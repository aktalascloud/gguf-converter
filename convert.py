"""Convert local model weights to GGUF through isolated CPU tools."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
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
    BACKENDS = HERE.parents[1] / 'gguf-q8'
PYTHON = Path(sys.executable).with_name('python.exe')
USER_DIRECTORY = Path(os.environ.get('GGUF_CONVERTER_CWD', os.getcwd()))
FORMATS = {'Q2': 'Q2_K', 'Q4': 'Q4_K', 'Q6': 'Q6_K', 'Q8': 'Q8_0'}
GPU_TARGETS = {
    'auto': 'Универсальный GGUF',
    'vulkan': 'Vulkan',
    'polaris': 'AMD Polaris / Vulkan',
    'cuda': 'NVIDIA CUDA / INT8',
    'pascal': 'NVIDIA CUDA / Pascal P40',
}
CPU_TARGETS = {'auto': 'Универсальный GGUF', 'avx': 'AVX', 'avx2': 'AVX2'}
GSQ_RCO = ('GSQ-RCO-Q2', 'GSQ-RCO-Q3', 'GSQ-RCO-Q4', 'GSQ-RCO-Q6', 'GSQ-RCO-Q8')
QUANTIZATIONS = (*FORMATS, *FORMATS.values(), 'Q4_0', *GSQ_RCO)
SAFETENSORS = {'.safetensors', '.sft'}
MODES = {
    'auto': 'Авто',
    'text': 'Текстовая модель',
    'diffusion': 'Фото / видео (SD, Flux, Wan и др.)',
    'moss-image': 'Диффузионные веса ComfyUI',
    'moss-text': 'Текстовый энкодер ComfyUI',
}
MEDIA_ARCHS = {
    'stable-diffusion', 'stable_diffusion', 'sd1', 'sd2', 'sd3', 'sdxl',
    'flux', 'flux1', 'flux2', 'flux3', 'wan', 'wan2', 'wan22', 'hunyuan_video',
    'ltxv', 'ltxav', 'qwen_image', 'qwen-image', 'z_image', 'z-image', 'moss_image',
}
VIDEO_IDS = ('wan', 'hunyuan_video', 'hunyuanvideo', 'ltx', 'mochi', 'cogvideo', 'skyreels', 'svd', 'animatediff')
PHOTO_IDS = ('stable_diffusion', 'stablediffusion', 'sdxl', 'flux', 'qwen_image', 'qwenimage',
             'z_image', 'zimage', 'pixart', 'sana', 'moss_image')
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


def output_quantization(quantization, gpu='auto', cpu='auto'):
    if gpu not in GPU_TARGETS or cpu not in CPU_TARGETS:
        raise ValueError('Неизвестная цель GPU или CPU.')
    if quantization in GSQ_RCO and (gpu != 'auto' or cpu != 'auto'):
        raise ValueError('GSQ/RCO выбирает типы по бюджету. Для этого режима оставьте цели GPU и CPU универсальными.')
    if quantization == 'Q4' and (gpu in ('polaris', 'pascal') or cpu != 'auto'):
        return 'Q4_0'
    return FORMATS.get(quantization, quantization)


def gguf_module(module):
    # Load the directory reader without the package's tokenizer and metadata extras.
    package = ModuleType('gguf')
    package.__path__ = [str(BACKENDS / 'hf' / 'gguf-py' / 'gguf')]
    sys.modules.setdefault(package.__name__, package)
    return import_module('gguf.' + module)


def open_gguf(path):
    reader = gguf_module('gguf_reader')
    reader.GGML_MAX_DIMS = max(reader.GGML_MAX_DIMS, 5)
    return reader.GGUFReader(path, 'r')


def read_gguf(path):
    reader = open_gguf(path)
    fields = {}
    for key in ('general.architecture', 'general.type', 'split.count', 'tokenizer.ggml.model'):
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


def media_role(architecture, names):
    tags = architecture.strip().lower().replace('-', '_')
    if any(tag in tags for tag in VIDEO_IDS):
        return 'Видео'
    names = {n.removeprefix('model.diffusion_model.').removeprefix('diffusion_model.') for n in names}
    if ('patch_embedding.weight' in names and any(n.startswith('blocks.0.self_attn.') for n in names)
            and any(n.startswith(('blocks.0.cross_attn.', 'time_embedding.')) for n in names)):
        return 'Видео'
    if tags in MEDIA_ARCHS or any(tag in tags for tag in PHOTO_IDS):
        return 'Фото'
    if any(n.startswith(('input_blocks.', 'double_blocks.', 'down_blocks.')) for n in names):
        return 'Фото'
    return ''


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
    label = media_role(arch, tensors)
    if mode == 'auto':
        media = bool(label) or any(n.startswith((
            'model.diffusion_model.', 'diffusion_model.', 'double_blocks.',
            'single_blocks.', 'input_blocks.', 'down_blocks.', 'transformer_blocks.',
        )) for n in tensors)
        if not arch and not media:
            if not fields.get('tokenizer.ggml.model'):
                raise ValueError('Сохраните GGUF с метаданными архитектуры или выберите исходные веса с конфигурацией.')
        mode = 'diffusion' if media else 'text'
        if mode == 'text' and arch and not fields.get('tokenizer.ggml.model'):
            mode = 'encoder'
    if mode == 'diffusion' and (fields.get('split.count', 0) > 1 or match):
        raise ValueError('Для диффузионной модели выберите единый GGUF. Разделённый GGUF поддерживается для текстовых моделей.')
    if mode == 'diffusion' and any(t == 'I8' or (t.startswith('F8_') and n + '_scale' in tensors)
                                   for n, t in tensors.items()):
        raise ValueError('Для медиа GGUF со шкалами INT8/FP8 выберите исходный Safetensors с метаданными квантования.')
    return source, mode, (label or 'Диффузионная модель') if mode == 'diffusion' else (
        'Текстовый энкодер' if mode == 'encoder' else 'Текстовая модель')


def route(source, mode):
    if mode.startswith('moss-'):
        if not source.is_file() or source.suffix.lower() not in SAFETENSORS:
            raise ValueError('Для упакованных весов ComfyUI выберите файл Safetensors.')
        return source, mode, 'Фото' if mode == 'moss-image' else 'Текстовый энкодер'
    if source.is_dir() or source.suffix.lower() == '.bin':
        if mode == 'diffusion':
            raise ValueError('Для диффузии выберите файл GGUF, Safetensors или CKPT.')
        if source.is_dir() and mode == 'auto':
            configs = [source / 'model_index.json', source / 'config.json']
            tag = ' '.join(config_tag(p) for p in configs)
            if media_role(tag, ()):
                weights = [p for p in source.glob('*.safetensors') if p.is_file()]
                if len(weights) != 1:
                    raise ValueError('Выберите файл весов диффузора. Для папки нужна одна модель Safetensors.')
                return route(weights[0], 'auto')
        return hf_folder(source), 'hf', 'Текстовая модель'
    if source.suffix.lower() == '.gguf':
        return gguf_source(source, mode)
    if source.suffix.lower() in SAFETENSORS:
        if mode == 'text':
            return hf_folder(source), 'hf', 'Текстовая модель'
        header, metadata, start = safe_header(source)
        tag = ' '.join(str(metadata.get(k, '')) for k in ('modelspec.architecture', 'architecture', 'model_type'))
        tag += ' ' + config_tag(source.parent / 'config.json')
        label = media_role(tag, header)
        if mode == 'auto':
            text_weight = any(n.startswith(('model.layers.', 'model.embed_tokens.', 'transformer.h.')) for n in header)
            formats = comfy_formats(source, header, metadata, start)
            w4a8 = any(cfg.get('format') == 'asym_w4a8_int8' for cfg in formats.values())
            moss = 'moss' in tag.lower() or any(n.endswith('layers.0.proj.qkv.weight') for n in header)
            text_component = text_weight or any(n.startswith(('model.language_model.', 'text_encoders.')) for n in header)
            model_folder = (source.parent / 'config.json').is_file()
            if (moss or w4a8) and (text_component or label == 'Фото' or
                                   any(n.startswith('model.diffusion_model.') for n in header)):
                return route(source, 'moss-text' if text_component else 'moss-image')
            if not label and (text_weight or (model_folder and source.name.startswith(('model', 'pytorch_model')))):
                return hf_folder(source), 'hf', 'Текстовая модель'
        return source, 'diffusion', label or 'Диффузионная модель'
    if source.suffix.lower() == '.ckpt' and mode != 'text':
        return source, 'diffusion', 'Диффузионная модель'
    raise ValueError('Выберите GGUF, Safetensors, CKPT или папку Hugging Face.')


def config_tag(path):
    if not path.is_file():
        return ''
    details = json.loads(path.read_text(encoding='utf-8-sig'))
    return ' '.join(str(details.get(key, '')) for key in ('architectures', '_class_name', 'model_type'))


def validate_media(source, moss):
    if source.suffix.lower() not in SAFETENSORS:
        return False
    header, metadata, start = safe_header(source)
    if 'ss_network_module' in metadata:
        raise ValueError('Выберите веса полной модели. Этот файл содержит адаптер LoRA.')
    formats = comfy_formats(source, header, metadata, start)
    expand = False
    if not moss and any(n.endswith(PACKED_KEYS) or '.quant_state.' in n for n in header):
        raise ValueError('Этот формат упакованных весов не распознан. Проверьте конфигурацию модели и метаданные шкал.')
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
                raise ValueError('Для W4A8 не определена архитектура по метаданным и тензорам.')
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
        file_type = gguf_file_type(quantization)
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

    def assemble_gsq_rco(self, manifest, output, bits):
        """Assemble prepared variants or calibrate selected tensors on Vulkan."""
        if manifest.suffix.lower() != '.json' or not manifest.is_file():
            raise ValueError('Для сборки GSQ-RCO выберите JSON-манифест оптимизированных тензоров.')
        def unique_pairs(pairs):
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ValueError('В манифесте GSQ-RCO повторяется ключ.')
            return result
        plan = json.loads(manifest.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_pairs)
        if not isinstance(plan, dict) or plan.get('format') not in ('gguf-gsq-rco-v1', 'gguf-gsq-rco-vulkan-v1'):
            raise ValueError('Ожидается манифест gguf-gsq-rco-v1 или gguf-gsq-rco-vulkan-v1.')
        targets = plan.get('targets')
        target = targets.get(str(bits)) if isinstance(targets, dict) else None
        if not isinstance(target, dict) or not isinstance(target.get('tensors'), dict) or not target['tensors']:
            raise ValueError(f'В манифесте отсутствуют назначения тензоров для Q{bits}.')
        def local_gguf(relative):
            if not isinstance(relative, str):
                raise ValueError('Варианты GSQ должны указывать на локальные файлы GGUF.')
            source = (manifest.parent / relative).resolve(strict=True)
            if source.suffix.lower() != '.gguf' or not source.is_file():
                raise ValueError(f'Ожидается файл GGUF: {source.name}')
            reader = open_gguf(source)
            if SHARD.fullmatch(source.name) or reader.get_field('split.count') is not None:
                raise ValueError('Для сборки GSQ-RCO используйте цельные GGUF.')
            return reader

        if plan['format'] == 'gguf-gsq-rco-vulkan-v1':
            sys.path.insert(0, str(HERE))
            try:
                from vulkan_workflow import optimize
                return optimize(self, manifest, local_gguf(plan.get('base')), target, output, gguf_module, bits,
                                capture=plan.get('capture'), detect=route, model_loss=plan.get('model_loss', False))
            finally:
                sys.path.pop(0)

        assignments = target['tensors']
        variants = plan.get('variants')
        if not isinstance(variants, dict) or any(not isinstance(v, str) or v not in variants for v in assignments.values()):
            raise ValueError('Назначения должны ссылаться на варианты из манифеста.')
        base = local_gguf(plan.get('base'))
        architecture = base.get_field('general.architecture')
        if (architecture is None or architecture.contents() in MEDIA_ARCHS
                or base.get_field('tokenizer.ggml.model') is None):
            raise ValueError('В режиме GSQ-RCO нужен GGUF текстовой модели с токенизатором.')
        model_keys = {'general.architecture', 'general.type', 'tokenizer.ggml.model',
                      'tokenizer.ggml.tokens', 'tokenizer.ggml.merges'}
        model_keys.update(key for key in base.fields if key.startswith(architecture.contents() + '.'))
        base_tensors = {t.name: t for t in base.tensors}
        if assignments.keys() - base_tensors.keys():
            raise ValueError('В назначениях RCO есть тензоры, которых нет в базовом GGUF.')
        chosen = base_tensors.copy()
        for label in sorted(set(assignments.values())):
            variant = local_gguf(variants[label])
            for key in sorted(model_keys):
                original, alternate = base.get_field(key), variant.get_field(key)
                if (original is None) != (alternate is None) or (original is not None and
                        (original.types != alternate.types or original.contents() != alternate.contents())):
                    raise ValueError(f'Вариант {label} не совпадает с базовой моделью по {key}.')
            if variant.endianess != base.endianess:
                raise ValueError('У базового файла и вариантов должен совпадать порядок байтов.')
            tensors = {t.name: t for t in variant.tensors}
            for name, selected in assignments.items():
                if selected != label:
                    continue
                candidate = tensors.get(name)
                if candidate is None or tuple(candidate.shape) != tuple(base_tensors[name].shape):
                    raise ValueError(f'Вариант {label} не содержит тензор нужной формы: {name}')
                if candidate.tensor_type.name in PLAIN_TYPES | {'I8'}:
                    raise ValueError(f'Для {name} нужен квантованный тензор GSQ в формате GGUF.')
                chosen[name] = candidate
        counts = Counter(t.tensor_type.name for t in chosen.values())
        if len(counts.keys() - PLAIN_TYPES - {'I8'}) < 2:
            raise ValueError('Для сборки GSQ-RCO ожидаются минимум два типа квантованных тензоров.')
        size = sum(t.n_bytes for t in chosen.values())
        elements = sum(t.n_elements for t in chosen.values())
        budget = target.get('max_tensor_bytes')
        if budget is not None and (type(budget) is not int or budget < size):
            raise ValueError(f'Объём тензоров {size} байт превышает бюджет манифеста или бюджет задан неверно.')
        if not elements:
            raise ValueError('Базовый GGUF не содержит весов.')
        replacements = {name: (chosen[name].data, chosen[name].tensor_type) for name in assignments}
        return self.write_mixed_gguf(base, output, replacements)

    def write_mixed_gguf(self, base, output, replacements):
        architecture = base.get_field('general.architecture')
        if architecture is None or not architecture.contents():
            raise ValueError('В базовом GGUF отсутствует general.architecture.')
        chosen = {t.name: replacements.get(t.name, (t.data, t.tensor_type)) for t in base.tensors}
        elements = sum(t.n_elements for t in base.tensors)
        size = sum(data.nbytes for data, _ in chosen.values())
        counts = Counter(kind.name for _, kind in chosen.values())
        with tempfile.TemporaryDirectory(prefix='.gguf-gsq-rco-', dir=output.parent) as temporary:
            stage = Path(temporary) / 'result.gguf'
            writer = gguf_module('gguf_writer').GGUFWriter(stage, architecture.contents(), endianess=base.endianess)
            try:
                for key, field in base.fields.items():
                    if key.startswith(('GGUF.', 'split.')) or key in ('general.architecture', 'general.file_type'):
                        continue
                    if key == 'general.alignment':
                        writer.add_custom_alignment(field.contents())
                    elif field.types:
                        writer.add_key_value(key, field.contents(), field.types[0],
                                             field.types[1] if len(field.types) > 1 else None)
                for name, (data, kind) in chosen.items():
                    writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, kind)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_ti_data_to_file()
                for data, _ in chosen.values():
                    if self.cancelled:
                        raise InterruptedError('Конвертация отменена.')
                    writer.write_tensor_data(data, tensor_endianess=base.endianess)
            finally:
                writer.close()
            _, actual_types = read_gguf(stage)
            if actual_types != {name: kind.name for name, (_, kind) in chosen.items()}:
                raise ValueError('Распределение типов в выходном GGUF не совпало с манифестом.')
            with self.lock:
                if self.cancelled:
                    raise InterruptedError('Конвертация отменена.')
                stage.rename(output)
        self.log(f'Готово: {output}\nТензоры: {dict(counts)}. Весовой объём: {8*size/elements:.3f} бит/параметр.')
        return output

    def convert(self, source, output=None, mode='auto', threads=None, quantization='Q8_0', gpu='auto', cpu='auto'):
        if quantization not in QUANTIZATIONS:
            raise ValueError(f'Неизвестный формат GGUF {quantization}.')
        quantization = output_quantization(quantization, gpu, cpu)
        source = Path(os.path.abspath(USER_DIRECTORY / Path(source).expanduser()))
        if not source.exists():
            raise FileNotFoundError(source)
        output = USER_DIRECTORY / Path(output).expanduser() if output else suggested_output(source, quantization)
        if output.suffix.lower() != '.gguf':
            raise ValueError('Имя результата должно заканчиваться на .gguf.')
        output = output.parent.resolve(strict=True) / output.name
        if os.path.lexists(output):
            raise FileExistsError(f'Файл уже существует. Выберите другое имя: {output}')
        if quantization in GSQ_RCO:
            return self.assemble_gsq_rco(source, output, int(quantization[-1]))
        source, mode, detected = route(source, mode)
        expand = False
        if mode in ('diffusion', 'moss-image', 'moss-text'):
            expand = validate_media(source, mode.startswith('moss-'))
        threads = threads if threads is not None else min(8, os.cpu_count() or 1)
        if threads < 1:
            raise ValueError('Число потоков должно быть положительным.')
        self.log(f'Распознано: {detected}. Формат {quantization}. GPU {GPU_TARGETS[gpu]}, CPU {CPU_TARGETS[cpu]}. Вход {source.name}')
        with tempfile.TemporaryDirectory(prefix='.gguf-convert-', dir=output.parent) as temp:
            stage = Path(temp) / 'result.gguf'
            llama = BACKENDS / 'llama' / 'llama-quantize.exe'
            llama_flags = ['--allow-requantize']
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
            elif mode == 'encoder' or (mode == 'diffusion' and source.suffix.lower() == '.gguf'):
                if SHARD.fullmatch(source.name) or read_gguf(source)[0].get('split.count', 0) > 1:
                    raise ValueError('Для отдельного текстового энкодера выберите единый GGUF.')
                self.requantize_gguf(source, stage, quantization, threads)
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
            if not counts[quantization] or raw_weights or set(counts) - OUTPUT_TYPES:
                raise ValueError(f'В выходном GGUF нет совместимых весов {quantization} или есть неподдерживаемые типы: {dict(counts)}')
            with self.lock:
                if self.cancelled:
                    raise InterruptedError('Конвертация отменена.')
                # Windows rename publishes the verified file atomically and fails if it already exists.
                stage.rename(output)
        self.log(f'Готово: {output}\nРазмер: {output.stat().st_size:,} байт ({output.stat().st_size / 2**30:.3f} GiB)\nТензоры: {dict(counts)}')
        return output

    def requantize_gguf(self, source, output, quantization, threads):
        import numpy as np

        reader = open_gguf(source)
        constants = gguf_module('constants')
        kind = constants.GGMLQuantizationType[quantization]
        block, block_bytes = constants.GGML_QUANT_SIZES[kind]
        dequantize = gguf_module('quants').dequantize
        encode = native_quantizer(kind)
        replacements = {}
        with ExitStack() as maps, ThreadPoolExecutor(max_workers=threads) as pool:
            for i, tensor in enumerate(reader.tensors):
                if self.cancelled:
                    raise InterruptedError('Конвертация отменена.')
                if len(tensor.shape) != 2 or not tensor.name.endswith('.weight') or tensor.tensor_type == kind:
                    continue
                width, rows = map(int, tensor.shape)
                if width % block:
                    continue
                data = np.memmap(output.parent / f'encoded-{i}.bin', mode='w+', dtype=np.uint8,
                                 shape=(rows, width // block * block_bytes))
                maps.callback(data._mmap.close)
                batch = max(1, (1024 * 1024) // width)

                def chunk(start):
                    if self.cancelled:
                        raise InterruptedError('Конвертация отменена.')
                    part = slice(start, start + batch)
                    encode(dequantize(tensor.data[part], tensor.tensor_type), data[part])

                for _ in pool.map(chunk, range(0, rows, batch)):
                    pass
                replacements[tensor.name] = data, kind
            return self.write_mixed_gguf(reader, output, replacements)


PLAIN_TYPES = {'F16', 'F32', 'BF16', 'F64', 'I16', 'I32', 'I64'}
OUTPUT_TYPES = PLAIN_TYPES | set(FORMATS.values()) | {'Q3_K', 'Q4_0', 'Q5_K'}


def gguf_file_type(quantization):
    name = 'Q4_K_M' if quantization == 'Q4_K' else quantization
    return getattr(gguf_module('constants').LlamaFileType, 'MOSTLY_' + name)


def native_quantizer(kind):
    """Encode standard GGUF rows with the same GGML library as llama-quantize."""
    import ctypes
    import numpy as np

    folder = BACKENDS / 'llama'
    with os.add_dll_directory(str(folder.resolve())):
        library = ctypes.CDLL(str((folder / 'ggml-base.dll').resolve()))
    encode = library.ggml_quantize_chunk
    encode.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p)
    encode.restype = ctypes.c_size_t

    def quantize(data, target):
        data = np.ascontiguousarray(data, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError('В весах есть Inf/NaN.')
        if encode(int(kind), data.ctypes.data, target.ctypes.data, 0, *data.shape, None) != target.nbytes:
            raise RuntimeError('GGML вернула неверный размер квантованных весов.')

    return quantize


def automatic_gsq_plan(source, tier, prompt, profile):
    reader = open_gguf(source)
    _, _, role = route(source, 'auto')
    if role not in ('Фото', 'Видео', 'Диффузионная модель'):
        raise ValueError('Для автозаписи нужен GGUF диффузора. Для энкодера или LLM доступны манифесты.')
    # ponytail: start with two attention projections; use a manifest to choose other matrices.
    choices = [t for t in reader.tensors if len(t.shape) == 2 and int(t.shape[0]) % 256 == 0
               and re.search(r'(?:self_attn\.[qkv]|proj\.qkv|proj\.o|attn[12]\.to_[qkv]|attn_qkv|q_proj|qkv)\.weight$', t.name)]
    if not choices:
        raise ValueError('Проекции для автозаписи не определены. Укажите тензоры в манифесте GSQ/RCO.')
    choices = choices[:2]
    kind = gguf_module('constants')
    qtype = kind.GGMLQuantizationType[FORMATS[f'Q{tier}']]
    block, size = kind.GGML_QUANT_SIZES[qtype]
    budget = sum(t.n_bytes for t in reader.tensors) + sum(t.n_elements // block * size - t.n_bytes for t in choices)
    capture = {'prompt': prompt}
    if profile:
        parent = profile.parent
        capture.update(profile=str(profile), profile_root=str(parent.parent if parent.name == 'profiles' else parent))
    return {'format': 'gguf-gsq-rco-vulkan-v1', 'base': str(source), 'model_loss': True, 'capture': capture,
            'targets': {str(tier): {'max_tensor_bytes': budget, 'tensors': {t.name: None for t in choices}}}}


def gui(source=''):
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
    from tkinter.scrolledtext import ScrolledText

    window = tk.Tk()
    window.title('Конвертер GGUF')
    window.geometry('800x480')
    window.minsize(600, 360)
    window.columnconfigure(1, weight=1)
    window.rowconfigure(7, weight=1)
    input_value = tk.StringVar(value=source)
    output_value = tk.StringVar()
    kind_value = tk.StringVar(value='Автоматически')
    quantization_value = tk.StringVar(value='Q8')
    optimize_value = tk.BooleanVar(value=False)
    gpu_value = tk.StringVar(value=GPU_TARGETS['auto'])
    cpu_value = tk.StringVar(value=CPU_TARGETS['auto'])
    format_value = tk.StringVar()
    guessed = ''
    events = queue.Queue()
    job = None
    closing = False

    def selection():
        value = quantization_value.get()
        return 'GSQ-RCO-' + value if optimize_value.get() else value

    def targets():
        if optimize_value.get():
            return 'auto', 'auto'
        gpu = next(k for k, label in GPU_TARGETS.items() if label == gpu_value.get())
        cpu = next(k for k, label in CPU_TARGETS.items() if label == cpu_value.get())
        return gpu, cpu

    def refresh(*_):
        nonlocal guessed
        state = 'disabled' if optimize_value.get() else 'readonly'
        gpu_box.config(state=state)
        cpu_box.config(state=state)
        resolved = output_quantization(selection(), *targets())
        format_value.set(resolved)
        value = input_value.get().strip().strip('"')
        path = USER_DIRECTORY / Path(value).expanduser()
        if not value or not path.exists():
            kind_value.set('Автоматически')
            return
        if path.suffix.lower() == '.json':
            kind_value.set('Манифест GSQ/RCO')
        else:
            try:
                kind_value.set(route(path, 'auto')[2])
            except (ValueError, OSError, KeyError, RuntimeError):
                kind_value.set('Требуется проверка файла')
        suggested = str(suggested_output(path, resolved))
        if not output_value.get() or output_value.get() == guessed:
            output_value.set(suggested)
        guessed = suggested

    def select_input(folder=False):
        value = filedialog.askdirectory(initialdir=USER_DIRECTORY) if folder else filedialog.askopenfilename(initialdir=USER_DIRECTORY,
            filetypes=[('Модели и манифест GSQ-RCO', '*.gguf *.safetensors *.sft *.ckpt *.bin *.json'), ('Все файлы', '*.*')])
        if value:
            if Path(value).suffix.lower() == '.json':
                optimize_value.set(True)
            output_value.set('')
            input_value.set(value)
            refresh()

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
            messagebox.showerror('GGUF', 'Выберите файл или папку модели.')
            return
        quantization = selection()
        gpu, cpu = targets()
        automatic = optimize_value.get() and Path(source).suffix.lower() == '.gguf'
        prompt = None
        profile = None
        if automatic:
            prompt = simpledialog.askstring('GSQ/RCO', 'Промпт для калибровки', parent=window,
                                            initialvalue='A red flower by a window, soft natural light.')
            if not prompt:
                return
            if messagebox.askyesno('GSQ/RCO', 'Подключить энкодер, VAE и другие веса из профиля .kcpps?', parent=window):
                path = filedialog.askopenfilename(parent=window, initialdir=USER_DIRECTORY,
                                                  filetypes=[('Профиль KoboldCpp', '*.kcpps')])
                if not path:
                    return
                profile = Path(path).resolve(strict=True)
        elif optimize_value.get() and Path(source).suffix.lower() != '.json':
            messagebox.showerror('GSQ/RCO', 'Для калибровки выберите GGUF или JSON-манифест.')
            return
        job = Conversion(lambda line: events.put(('log', line)))
        start_button.config(state='disabled')
        cancel_button.config(state='normal')

        def work():
            try:
                if automatic:
                    original = (USER_DIRECTORY / Path(source).expanduser()).resolve(strict=True)
                    with tempfile.TemporaryDirectory(prefix='gguf-plan-') as temp:
                        plan = Path(temp) / 'capture.json'
                        payload = automatic_gsq_plan(original, int(quantization[-1]), prompt, profile)
                        plan.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
                        job.convert(plan, output or suggested_output(original, quantization), quantization=quantization)
                else:
                    job.convert(source, output, quantization=quantization, gpu=gpu, cpu=cpu)
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
        elif messagebox.askyesno('GGUF', 'Остановить текущую конвертацию?'):
            closing = True
            job.cancel()

    for row, label, value in ((0, 'Источник', input_value), (1, 'Результат', output_value)):
        ttk.Label(window, text=label).grid(row=row, column=0, sticky='w', padx=8, pady=5)
        field = ttk.Entry(window, textvariable=value)
        field.grid(row=row, column=1, sticky='ew', padx=4, pady=5)
        if row == 0:
            field.bind('<FocusOut>', refresh)
            field.bind('<Return>', refresh)
    ttk.Button(window, text='Файл…', command=select_input).grid(row=0, column=2, padx=4)
    ttk.Button(window, text='Папка…', command=lambda: select_input(True)).grid(row=0, column=3, padx=4)
    ttk.Button(window, text='Сохранить как…', command=select_output).grid(row=1, column=2, columnspan=2, sticky='ew', padx=4)
    ttk.Label(window, text='Тип модели').grid(row=2, column=0, sticky='w', padx=8)
    ttk.Label(window, textvariable=kind_value).grid(row=2, column=1, sticky='w', padx=4)
    ttk.Label(window, text='Формат').grid(row=3, column=0, sticky='w', padx=8)
    format_box = ttk.Combobox(window, textvariable=quantization_value, values=tuple(FORMATS), state='readonly', width=8)
    format_box.grid(row=3, column=1, sticky='w', padx=4)
    format_box.bind('<<ComboboxSelected>>', refresh)
    ttk.Label(window, textvariable=format_value).grid(row=2, column=2, columnspan=2, sticky='w', padx=4)
    ttk.Checkbutton(window, text='GSQ/RCO (Vulkan)', variable=optimize_value, command=refresh).grid(row=3, column=2, columnspan=2, sticky='w', padx=4)
    for row, label, var, choices in ((4, 'Цель GPU', gpu_value, GPU_TARGETS), (5, 'Цель CPU', cpu_value, CPU_TARGETS)):
        ttk.Label(window, text=label).grid(row=row, column=0, sticky='w', padx=8)
        control = ttk.Combobox(window, textvariable=var, values=list(choices.values()), state='readonly')
        control.grid(row=row, column=1, sticky='ew', padx=4)
        if row == 4:
            gpu_box = control
        else:
            cpu_box = control

    def select_gpu(event):
        if targets()[0] == 'cuda':
            quantization_value.set('Q8')
        refresh(event)
    gpu_box.bind('<<ComboboxSelected>>', select_gpu)
    cpu_box.bind('<<ComboboxSelected>>', refresh)
    start_button = ttk.Button(window, text='Конвертировать', command=start)
    start_button.grid(row=6, column=1, sticky='w', padx=4, pady=8)
    cancel_button = ttk.Button(window, text='Отмена', state='disabled', command=lambda: job.cancel())
    cancel_button.grid(row=6, column=2, columnspan=2, sticky='ew', padx=4)
    console = ScrolledText(window, wrap='word')
    console.grid(row=7, column=0, columnspan=4, sticky='nsew', padx=8, pady=(0, 8))
    window.protocol('WM_DELETE_WINDOW', close)
    if source and Path(source).suffix.lower() == '.json':
        optimize_value.set(True)
    refresh()
    poll()
    window.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Автоопределение модели и конвертация GGUF в Q2, Q4, Q6, Q8')
    parser.add_argument('source', nargs='?')
    parser.add_argument('-o', '--output')
    parser.add_argument('--mode', choices=MODES, default='auto')
    parser.add_argument('--gpu', choices=GPU_TARGETS, default='auto')
    parser.add_argument('--cpu', choices=CPU_TARGETS, default='auto')
    parser.add_argument('-t', '--threads', type=int)
    parser.add_argument('-q', '--quantization', type=str.upper, choices=QUANTIZATIONS, default='Q8_0')
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    try:
        if args.gui or args.source is None:
            gui(args.source or '')
        else:
            Conversion().convert(args.source, args.output, args.mode, args.threads, args.quantization, args.gpu, args.cpu)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Ошибка: {exc or "конвертация прервана"}', file=sys.stderr)
        sys.exit(1)
