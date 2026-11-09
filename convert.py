"""Convert local model weights to GGUF through isolated CPU tools."""

import argparse
from collections import Counter
from importlib import import_module
import json
from math import prod
import os
from pathlib import Path
import queue
import re
import struct
import subprocess
import sys
import tempfile
import threading
from types import ModuleType

HERE = Path(__file__).resolve().parent
BACKENDS = HERE / 'backends'
if not BACKENDS.is_dir():
    BACKENDS = HERE.parents[1] / 'build' / 'gguf-q8'
PYTHON = Path(sys.executable).with_name('python.exe')
USER_DIRECTORY = Path(os.environ.get('GGUF_CONVERTER_CWD', os.getcwd()))
QUANTIZATIONS = ('Q8_0', 'Q4_0')
SAFETENSORS = {'.safetensors', '.sft'}
MODES = {
    'auto': 'Авто',
    'text': 'Текстовая модель',
    'diffusion': 'Фото / видео (SD, Flux, Wan и др.)',
    'moss-image': 'Artius MOSS · диффузор',
    'moss-text': 'Artius MOSS · текстовый энкодер для SD',
}
MEDIA_ARCHS = {
    'stable-diffusion', 'stable_diffusion', 'sd1', 'sd2', 'sd3', 'sdxl',
    'flux', 'flux1', 'flux2', 'flux3', 'wan', 'wan2', 'wan22', 'hunyuan_video',
    'ltxv', 'ltxav', 'qwen_image', 'qwen-image', 'z_image', 'z-image', 'moss_image',
}
SHARD = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', re.IGNORECASE)
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
PACKED_KEYS = ('.qweight', '.qzeros', '_packed', '.weight.absmax', '.weight.quant_map')
SAFE_WIDTHS = {
    'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
    'F8_E4M3': 1, 'F8_E5M2': 1, 'F8_E8M0': 1,
    'I64': 8, 'U64': 8, 'I32': 4, 'U32': 4, 'I16': 2, 'U16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1,
}


def suggested_output(source, quantization='Q8_0'):
    source = USER_DIRECTORY / source
    suffix = source.suffix.lower()
    name = source.stem if source.is_file() or suffix in SAFETENSORS | {'.gguf', '.bin', '.ckpt'} else source.name
    if match := SHARD.fullmatch(source.name):
        name = match[1]
    return source.parent / f'{name}-{quantization}.gguf'


def gguf_module(module):
    # Load the directory reader without the package's tokenizer and metadata extras.
    package = ModuleType('gguf')
    package.__path__ = [str(BACKENDS / 'hf' / 'gguf-py' / 'gguf')]
    sys.modules.setdefault(package.__name__, package)
    return import_module('gguf.' + module)


def open_gguf(path):
    return gguf_module('gguf_reader').GGUFReader(path, 'r')


def read_gguf(path):
    reader = open_gguf(path)
    fields = {}
    for key in ('general.architecture', 'general.type', 'split.count'):
        field = reader.get_field(key)
        if field is not None:
            fields[key] = field.contents()
    types = {tensor.name: tensor.tensor_type.name for tensor in reader.tensors}
    return fields, types


def safe_header(source):
    size = source.stat().st_size
    with source.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('Неполный заголовок Safetensors.')
        length, = struct.unpack('<Q', raw)
        if not 2 <= length <= min(100_000_000, size - 8):
            raise ValueError('Недопустимая длина заголовка Safetensors.')
        header = json.loads(stream.read(length))
    if not isinstance(header, dict) or not header:
        raise ValueError('Пустой заголовок Safetensors.')
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict) or not header:
        raise ValueError('В Safetensors нет тензоров или повреждены метаданные.')
    ranges = []
    for name, info in header.items():
        start, end = info['data_offsets']
        shape = info['shape']
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= size - 8 - length):
            raise ValueError(f'Повреждён тензор {name}.')
        width = SAFE_WIDTHS.get(info['dtype'])
        if width is None or prod(shape) * width != end - start:
            raise ValueError(f'Неизвестный тип или неверный размер тензора {name}.')
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError('Повреждены диапазоны данных Safetensors.')
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError('Размер данных Safetensors не совпадает с заголовком.')
    return header, metadata, length + 8


def comfy_formats(source, header, metadata, start):
    formats = json.loads(metadata.get('_quantization_metadata', '{}')).get('layers', {}).copy()
    with source.open('rb') as stream:
        for key, info in header.items():
            if key.endswith('.comfy_quant'):
                a, b = info['data_offsets']
                if info['dtype'] != 'U8' or not 0 < b - a <= 65536:
                    raise ValueError(f'Повреждена схема квантования {key}.')
                stream.seek(start + a)
                formats[key.removesuffix('.comfy_quant')] = json.loads(stream.read(b - a))
    return formats


def hf_folder(source):
    folder = source if source.is_dir() else source.parent
    if not (folder / 'config.json').is_file():
        raise ValueError('Для текстовых весов выберите папку Hugging Face с config.json и токенизатором.')
    for name in ('model.safetensors', 'pytorch_model.bin'):
        index = folder / (name + '.index.json')
        if index.exists():
            parts = set(json.loads(index.read_text(encoding='utf-8-sig'))['weight_map'].values())
            for part in parts:
                if not isinstance(part, str) or not part or any(c in part for c in '/\\:') or part in {'.', '..'}:
                    raise ValueError('Индекс весов ссылается на файл за пределами папки модели.')
                if not (folder / part).is_file():
                    raise ValueError(f'Не найден шард {part}.')
        else:
            prefix, ext = name.split('.')
            parts = {p.name for p in folder.glob(f'{prefix}*.{ext}') if p.is_file()}
            if parts and parts != {name}:
                if len(parts) == 1 and name not in parts:
                    single = next(iter(parts))
                    if re.search(r'-\d+-of-\d+\.', single):
                        raise ValueError(f'Для шардов нужен {index.name}.')
                else:
                    raise ValueError(f'В папке несколько вариантов весов. Нужен {index.name} или один файл {name}.')
        if parts:
            if source.is_file() and source.name not in parts:
                raise ValueError('Выбранный файл не входит в используемый набор весов. Проверьте папку модели.')
            return folder
    raise ValueError('В папке нет весов model*.safetensors или pytorch_model*.bin.')


def gguf_source(source, mode):
    if match := SHARD.fullmatch(source.name):
        stem, _, total = match.groups()
        if not 1 <= int(total) <= 1000:
            raise ValueError('Недопустимое число шардов GGUF.')
        for index in range(1, int(total) + 1):
            part = source.with_name(f'{stem}-{index:05d}-of-{total}.gguf')
            if not part.is_file():
                raise ValueError(f'Не найден шард {part.name}.')
            if index == 1:
                first = part
        source = first
    fields, tensors = read_gguf(source)
    if fields.get('general.type') == 'adapter':
        raise ValueError('Выберите веса полной модели. Этот GGUF содержит адаптер.')
    arch = fields.get('general.architecture', '')
    if mode == 'auto':
        media = arch in MEDIA_ARCHS or any(n.startswith((
            'model.diffusion_model.', 'diffusion_model.', 'double_blocks.',
            'single_blocks.', 'input_blocks.', 'down_blocks.', 'transformer_blocks.',
        )) for n in tensors)
        if not arch and not media:
            raise ValueError('В GGUF не указана архитектура. Выберите режим модели вручную.')
        mode = 'diffusion' if media else 'text'
    if mode == 'diffusion' and (fields.get('split.count', 0) > 1 or match):
        raise ValueError('Для диффузионной модели выберите единый GGUF. Разделённый GGUF поддерживается для текстовых моделей.')
    if mode == 'diffusion' and any(t == 'I8' or (t.startswith('F8_') and n + '_scale' in tensors)
                                   for n, t in tensors.items()):
        raise ValueError('Для медиа GGUF со шкалами INT8/FP8 выберите исходный Safetensors с метаданными квантования.')
    return source, mode


def route(source, mode):
    if mode.startswith('moss-'):
        if not source.is_file() or source.suffix.lower() not in SAFETENSORS:
            raise ValueError('Для Artius MOSS выберите файл Safetensors.')
        return source, mode
    if source.is_dir() or source.suffix.lower() == '.bin':
        if mode == 'diffusion':
            raise ValueError('Для диффузии выберите файл GGUF, Safetensors или CKPT.')
        return hf_folder(source), 'hf'
    if source.suffix.lower() == '.gguf':
        return gguf_source(source, mode)
    if source.suffix.lower() in SAFETENSORS:
        if mode == 'text':
            return hf_folder(source), 'hf'
        if mode == 'auto':
            header, _, _ = safe_header(source)
            text_weight = any(n.startswith(('model.layers.', 'model.embed_tokens.', 'transformer.h.')) for n in header)
            config_path = source.parent / 'config.json'
            config = json.loads(config_path.read_text(encoding='utf-8-sig')) if config_path.is_file() else {}
            if text_weight or (source.name.startswith('model') and (config.get('architectures') or config.get('model_type'))):
                return hf_folder(source), 'hf'
        return source, 'diffusion'
    if source.suffix.lower() == '.ckpt' and mode != 'text':
        return source, 'diffusion'
    raise ValueError('Выберите GGUF, Safetensors, CKPT или папку Hugging Face.')


def validate_media(source, moss):
    if source.suffix.lower() not in SAFETENSORS:
        return False
    header, metadata, start = safe_header(source)
    if 'ss_network_module' in metadata:
        raise ValueError('Выберите веса полной модели. Этот файл содержит адаптер LoRA.')
    formats = comfy_formats(source, header, metadata, start)
    expand = False
    if not moss and any(n.endswith(PACKED_KEYS) or '.quant_state.' in n for n in header):
        raise ValueError('Обнаружены упакованные веса. Для текстовой модели выберите папку Hugging Face, для W4A8 Artius выберите режим MOSS.')
    for name, info in header.items():
        if not name.endswith('.weight'):
            continue
        base = name.removesuffix('.weight')
        cfg = formats.get(base, {})
        scheme = cfg.get('format', '')
        if info['dtype'] in {'I8', 'F8_E5M2'} or (info['dtype'] == 'F8_E4M3' and (
                scheme or any(base + suffix in header for suffix in ('.weight_scale', '.weight_scale_inv', '.scale_weight')))):
            expand = True
        if scheme == 'asym_w4a8_int8':
            if not moss:
                raise ValueError('Для W4A8 Artius выберите режим MOSS.')
            if (len(info['shape']) != 2 or info['shape'][1] % 16 or not all(
                    base + suffix in header for suffix in ('.weight_codebook', '.weight_s_rel', '.weight_s_channel'))):
                raise ValueError(f'Неполная упаковка W4A8 у {name}.')
        elif info['dtype'] in {'I8', 'U8', 'I32', 'I16'}:
            if scheme != 'int8_tensorwise' or info['dtype'] != 'I8' or base + '.weight_scale' not in header:
                raise ValueError(f'Для целочисленного веса {name} нужна поддерживаемая схема и масштаб.')
        if moss and info['dtype'] in {'I8', 'U8'}:
            stride = 16 if scheme == 'asym_w4a8_int8' else 32
            if len(info['shape']) != 2 or info['shape'][1] % stride:
                raise ValueError(f'Форма целочисленного веса {name} несовместима с декодером MOSS.')
        if moss and info['dtype'] == 'F8_E4M3':
            scale = header.get(base + '.weight_scale')
            if base + '.scale_weight' in header or base + '.weight_scale_inv' in header:
                raise ValueError(f'В режиме MOSS у {name} нужен масштаб weight_scale.')
            if scale and (len(info['shape']) != 2 or info['shape'][1] % 32):
                raise ValueError(f'Форма масштабированного веса {name} несовместима с режимом MOSS.')
        if moss and scheme not in {'', 'int8_tensorwise', 'asym_w4a8_int8', 'float8_e4m3fn'}:
            raise ValueError(f'Декодер MOSS не поддерживает {scheme}.')
    return expand


class Conversion:
    def __init__(self, log=print):
        self.log = log
        self.lock = threading.Lock()
        self.cancelled = False
        self.child = None

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()

    def run(self, executable, *args, cwd=None, threads=None):
        executable = Path(executable)
        if not executable.is_file():
            raise FileNotFoundError(f'Не установлен конвертер {executable.name}. См. README.md.')
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1')
        if threads is not None:
            env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        if cwd is not None:
            env.update(TEMP=str(cwd), TMP=str(cwd))
        env.pop('NO_LOCAL_GGUF', None)
        with self.lock:
            if self.cancelled:
                raise InterruptedError('Конвертация отменена.')
            self.child = child = subprocess.Popen(
                [str(executable), *map(str, args)], cwd=cwd or executable.parent, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                errors='replace', creationflags=subprocess.CREATE_NO_WINDOW,
            )
        try:
            with child.stdout:
                for line in child.stdout:
                    if line.strip():
                        self.log(ANSI.sub('', line.rstrip()))
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                child.terminate()
            child.wait()
            raise
        finally:
            with self.lock:
                self.child = None
        if self.cancelled:
            raise InterruptedError('Конвертация отменена.')
        if code:
            raise RuntimeError(f'{executable.name} завершился с кодом {code}. Подробности выше.')

    def preserve_media_arch(self, source, stage, quantization):
        arch = None
        if source.suffix.lower() == '.gguf':
            arch = read_gguf(source)[0].get('general.architecture')
        reader = open_gguf(stage)
        if not arch:
            names = {t.name.removeprefix('model.diffusion_model.').removeprefix('diffusion_model.')
                     for t in reader.tensors}
            if {'blocks.0.cross_attn.norm_k.weight', 'patch_embedding.weight'} <= names:
                arch = 'wan'
        field = reader.get_field('general.architecture')
        if not arch or (field is not None and field.contents() == arch):
            return stage
        self.log(f'Записываю архитектуру GGUF {arch}.')
        result = stage.with_name('with-architecture.gguf')
        writer = import_module('gguf.gguf_writer').GGUFWriter(result, arch)
        file_type = getattr(gguf_module('constants').LlamaFileType, 'MOSTLY_' + quantization)
        try:
            writer.add_name(source.stem)
            writer.add_file_type(file_type)
            for tensor in reader.tensors:
                writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                                       tensor.n_bytes, tensor.tensor_type)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            for tensor in reader.tensors:
                if self.cancelled:
                    raise InterruptedError('Конвертация отменена.')
                writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        finally:
            writer.close()
        return result

    def convert(self, source, output=None, mode='auto', threads=None, quantization='Q8_0'):
        if quantization not in QUANTIZATIONS:
            raise ValueError(f'Неизвестный формат GGUF {quantization}.')
        source = Path(os.path.abspath(USER_DIRECTORY / Path(source).expanduser()))
        if not source.exists():
            raise FileNotFoundError(source)
        output = USER_DIRECTORY / Path(output).expanduser() if output else suggested_output(source, quantization)
        if output.suffix.lower() != '.gguf':
            raise ValueError('Имя результата должно заканчиваться на .gguf.')
        output = output.parent.resolve(strict=True) / output.name
        if os.path.lexists(output):
            raise FileExistsError(f'Файл уже существует. Выберите другое имя: {output}')
        source, mode = route(source, mode)
        expand = False
        if mode in ('diffusion', 'moss-image', 'moss-text'):
            expand = validate_media(source, mode.startswith('moss-'))
        threads = threads if threads is not None else min(8, os.cpu_count() or 1)
        if threads < 1:
            raise ValueError('Число потоков должно быть положительным.')
        self.log(f'Режим {mode}. Формат {quantization}. Вход {source}')
        with tempfile.TemporaryDirectory(prefix='.gguf-convert-', dir=output.parent) as temp:
            stage = Path(temp) / 'result.gguf'
            llama = BACKENDS / 'llama' / 'llama-quantize.exe'
            llama_flags = ['--allow-requantize']
            if quantization == 'Q4_0':
                llama_flags.append('--pure')
            if mode == 'hf':
                hf_type = 'q8_0' if quantization == 'Q8_0' else 'f32'
                fp8_args = ['--fp8-as-q8'] if quantization == 'Q8_0' else []
                self.run(PYTHON, '-I', '-B', '-X', 'utf8', HERE / 'hf.py', source,
                         '--outfile', stage, '--outtype', hf_type, *fp8_args,
                         '--use-temp-file', cwd=temp, threads=threads)
                _, types = read_gguf(stage)
                if hf_type == 'f32' or any(t not in PLAIN_TYPES | {quantization} for t in types.values()):
                    final = stage.with_name('requantized.gguf')
                    self.run(llama, *llama_flags, stage, final, quantization, threads)
                    stage = final
            elif mode == 'text':
                self.run(llama, *llama_flags, source, stage, quantization, threads)
            elif mode == 'diffusion':
                original = source
                if expand:
                    restored = stage.with_name('decoded.safetensors')
                    self.run(PYTHON, '-I', '-B', '-X', 'utf8',
                             HERE / 'media_weights.py', source, restored)
                    source = restored
                self.run(BACKENDS / 'sd' / 'sd-cli.exe', '-M', 'convert', '-m', source,
                         '-o', stage, '--type', quantization.lower(), '-t', threads)
                stage = self.preserve_media_arch(original, stage, quantization)
            else:
                self.run(PYTHON, '-I', '-B', '-X', 'utf8', HERE / 'moss_convert.py',
                         source, stage, mode.removeprefix('moss-'), quantization)
            _, types = read_gguf(stage)
            counts = Counter(types.values())
            raw_weights = any(name.endswith('.weight') and kind.startswith('I') for name, kind in types.items())
            if not counts[quantization] or raw_weights or set(counts) - PLAIN_TYPES - {quantization}:
                raise ValueError(f'Конвертер вернул GGUF без целевой схемы {quantization}: {dict(counts)}')
            with self.lock:
                if self.cancelled:
                    raise InterruptedError('Конвертация отменена.')
                # Windows rename publishes the verified file atomically and fails if it already exists.
                stage.rename(output)
        self.log(f'Готово: {output}\nТензоры: {dict(counts)}')
        return output


PLAIN_TYPES = {'F16', 'F32', 'BF16', 'F64', 'I16', 'I32', 'I64'}


def gui(source=''):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    window = tk.Tk()
    window.title('Конвертер GGUF')
    window.geometry('800x480')
    window.minsize(600, 360)
    window.columnconfigure(1, weight=1)
    window.rowconfigure(5, weight=1)
    input_value = tk.StringVar(value=source)
    output_value = tk.StringVar()
    mode_value = tk.StringVar(value=MODES['auto'])
    quantization_value = tk.StringVar(value=QUANTIZATIONS[0])
    output_value.set(str(suggested_output(Path(source))) if source else '')
    events = queue.Queue()
    job = None
    closing = False

    def select_input(folder=False):
        value = filedialog.askdirectory(initialdir=USER_DIRECTORY) if folder else filedialog.askopenfilename(initialdir=USER_DIRECTORY,
            filetypes=[('Файлы моделей', '*.gguf *.safetensors *.sft *.ckpt *.bin'), ('Все файлы', '*.*')])
        if value:
            input_value.set(value)
            output_value.set(str(suggested_output(Path(value), quantization_value.get())))

    def update_extension(event):
        old = Path(output_value.get())
        if not old.name:
            return
        for quant in QUANTIZATIONS:
            if old.name.endswith('-' + quant + '.gguf'):
                output_value.set(str(old.with_name(old.name.removesuffix(quant + '.gguf') + quantization_value.get() + '.gguf')))
                break

    def select_output():
        value = filedialog.asksaveasfilename(defaultextension='.gguf', filetypes=[('GGUF', '*.gguf')],
                                             initialfile=Path(output_value.get()).name)
        if value:
            output_value.set(value)

    def start():
        nonlocal job
        source = input_value.get().strip().strip('"')
        output = output_value.get().strip().strip('"') or None
        if not source:
            messagebox.showerror('GGUF Q8_0', 'Выберите файл или папку модели.')
            return
        mode = next(key for key, label in MODES.items() if label == mode_value.get())
        quantization = quantization_value.get()
        job = Conversion(lambda line: events.put(('log', line)))
        start_button.config(state='disabled')
        cancel_button.config(state='normal')

        def work():
            try:
                job.convert(source, output, mode, quantization=quantization)
            except Exception as exc:
                events.put(('log', f'Ошибка: {exc}'))
            finally:
                events.put(('done', ''))

        threading.Thread(target=work).start()

    def poll():
        nonlocal job
        for _ in range(150):
            try:
                kind, value = events.get_nowait()
            except queue.Empty:
                break
            if kind == 'done':
                job = None
                if closing:
                    window.destroy()
                    return
                start_button.config(state='normal')
                cancel_button.config(state='disabled')
            else:
                console.insert('end', value + '\n')
                console.see('end')
        window.after(80, poll)

    def close():
        nonlocal closing
        if job is None:
            window.destroy()
        elif messagebox.askyesno('GGUF Q8_0', 'Остановить текущую конвертацию?'):
            closing = True
            job.cancel()

    for row, label, value in ((0, 'Источник', input_value), (1, 'Результат', output_value)):
        ttk.Label(window, text=label).grid(row=row, column=0, sticky='w', padx=8, pady=5)
        ttk.Entry(window, textvariable=value).grid(row=row, column=1, sticky='ew', padx=4, pady=5)
    ttk.Button(window, text='Файл…', command=select_input).grid(row=0, column=2, padx=4)
    ttk.Button(window, text='Папка…', command=lambda: select_input(True)).grid(row=0, column=3, padx=4)
    ttk.Button(window, text='Сохранить как…', command=select_output).grid(row=1, column=2, columnspan=2, sticky='ew', padx=4)
    ttk.Label(window, text='Режим').grid(row=2, column=0, sticky='w', padx=8)
    ttk.Combobox(window, textvariable=mode_value, values=list(MODES.values()), state='readonly').grid(row=2, column=1, sticky='ew', padx=4)
    ttk.Label(window, text='Формат').grid(row=3, column=0, sticky='w', padx=8)
    format_box = ttk.Combobox(window, textvariable=quantization_value, values=QUANTIZATIONS, state='readonly')
    format_box.grid(row=3, column=1, sticky='w', padx=4)
    format_box.bind('<<ComboboxSelected>>', update_extension)
    start_button = ttk.Button(window, text='Конвертировать', command=start)
    start_button.grid(row=4, column=1, sticky='w', padx=4, pady=8)
    cancel_button = ttk.Button(window, text='Отмена', state='disabled', command=lambda: job.cancel())
    cancel_button.grid(row=4, column=2, columnspan=2, sticky='ew', padx=4)
    console = ScrolledText(window, wrap='word')
    console.grid(row=5, column=0, columnspan=4, sticky='nsew', padx=8, pady=(0, 8))
    window.protocol('WM_DELETE_WINDOW', close)
    poll()
    window.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Преобразование моделей в GGUF Q8_0 или Q4_0')
    parser.add_argument('source', nargs='?')
    parser.add_argument('-o', '--output')
    parser.add_argument('--mode', choices=MODES, default='auto')
    parser.add_argument('-t', '--threads', type=int)
    parser.add_argument('-q', '--quantization', type=str.upper, choices=QUANTIZATIONS, default='Q8_0')
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    try:
        if args.gui or args.source is None:
            gui(args.source or '')
        else:
            Conversion().convert(args.source, args.output, args.mode, args.threads, args.quantization)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Ошибка: {exc or "конвертация прервана"}', file=sys.stderr)
        sys.exit(1)
