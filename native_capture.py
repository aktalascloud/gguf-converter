"""Collect activation rows from the bundled Vulkan media runtime."""
from contextlib import contextmanager
import base64
import io
import json
import os
from pathlib import Path
import re
import secrets
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np


BACKEND = Path(__file__).resolve().parent / 'capture-runtime'
FILE_KEYS = ('sdmodel', 'sdclip1', 'sdclip2', 'sdllm', 'sdvae')
PROFILE_OPTIONS = ('sdthreads', 'sdclipdevice', 'sdvaedevice', 'sdmaingpu', 'sdflashattention',
                   'sdconvdirect', 'sdoffloadcpu', 'sdvramlimit', 'sdvaetiled', 'sdvaeauto', 'sdtiledvae',
                   'sdclamped', 'sdclampedsoft', 'sdloramult')
TMP_LOG = 'native.log'


def runtime_config(recipe, manifest, reference):
    if not isinstance(recipe, dict):
        raise ValueError('Укажите настройки автозаписи активаций в объекте capture.')
    root = manifest.parent
    defaults = {}
    profile = recipe.get('profile')
    if profile:
        path = (root / profile).resolve(strict=True)
        defaults = json.loads(path.read_text(encoding='utf-8-sig'))
        root = (root / recipe.get('profile_root', str(path.parent))).resolve(strict=True)
    settings = {key: defaults[key] for key in PROFILE_OPTIONS if key in defaults}
    overrides = recipe.get('models', {})
    if not isinstance(overrides, dict) or overrides.keys() - set(FILE_KEYS):
        raise ValueError('capture.models принимает sdmodel, sdclip1, sdclip2, sdllm и sdvae.')
    part = recipe.get('base_part', 'sdmodel')
    if part not in FILE_KEYS:
        raise ValueError('Укажите, какую часть медиамодели заменяет исходный GGUF.')
    for key in FILE_KEYS:
        value = overrides.get(key, defaults.get(key))
        if key == part:
            value = str(reference)
        if value:
            resolved = (root / value).resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f'{key} должен указывать на файл весов.')
            settings[key] = str(resolved)
    loras = defaults.get('sdlora') or []
    if not isinstance(loras, list) or not all(isinstance(path, str) for path in loras):
        raise ValueError('В профиле sdlora должен быть списком локальных файлов.')
    if loras:
        paths = [(root / item).resolve(strict=True) for item in loras]
        if not all(item.is_file() for item in paths):
            raise ValueError('LoRA из профиля должна указывать на файл весов.')
        settings['sdlora'] = [str(item) for item in paths]
    if 'sdmodel' not in settings:
        raise ValueError('Автозапись для энкодера требует диффузор sdmodel.')
    validate_name = recipe.get('prompt')
    if not isinstance(validate_name, str) or not 0 < len(validate_name.strip()) <= 2400:
        raise ValueError('Укажите промпт для калибровки в capture.prompt.')
    seed = recipe.get('seed', 1729)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Некорректный seed калибровки.')
    return settings, part, validate_name, seed


def render_spec(settings, prompt, seed, detect):
    _, _, label = detect(Path(settings['sdmodel']), 'auto')
    video = label == 'Видео'
    if label not in ('Фото', 'Видео', 'Диффузионная модель'):
        raise ValueError('Автозапись активаций поддерживает диффузионные модели и подключённые энкодеры.')
    return dict(prompt=prompt, negative_prompt='', seed=seed, width=320 if video else 256,
                height=192 if video else 384, frames=5 if video else 1,
                fps=24, steps=4 if video else 2, cfg_scale=1, sampler_name='euler',
                scheduler='simple', flow_shift=5, eta=0, n_iter=1, batch_size=1,
                video_output_type=1, enable_hr=False)


@contextmanager
def worker(converter, settings, directory, activations):
    script = BACKEND / 'koboldcpp.py'
    if not script.is_file() or not (BACKEND / 'koboldcpp_vulkan.dll').is_file():
        raise FileNotFoundError('В EXE нет компонента Vulkan для автозаписи активаций.')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    password = secrets.token_hex(16)
    profile = directory / 'owned.kcpps'
    profile.write_text(json.dumps(dict(settings, usevulkan=[0], gpulayers=99, host='127.0.0.1', port=port,
                                       password=password), ensure_ascii=False), encoding='utf-8')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('KCPP_', 'GGML_', 'CUDA_', 'HIP_', 'ROCR_'))}
    env.update(PYTHONUNBUFFERED='1', TEMP=str(directory), TMP=str(directory))
    if activations:
        env['KCPP_GGUF_ACTIVATIONS'] = '.'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None, timeout=300):
        req = urllib.request.Request(f'http://127.0.0.1:{port}' + path,
                                     None if payload is None else json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + password})
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    with (directory / TMP_LOG).open('wb') as log:
        with converter.lock:
            if converter.cancelled:
                raise InterruptedError('Конвертация отменена.')
            process = converter.child = subprocess.Popen(
                [sys.executable, '-I', '-B', '-X', 'utf8', str(script), '--config', str(profile), '--skiplauncher'],
                cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    version = request('/api/extra/version', timeout=2)
                    if version.get('txt2img') and process.poll() is None:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.4)
            else:
                tail = (directory / TMP_LOG).read_bytes()[-2400:].decode('utf-8', errors='replace')
                raise RuntimeError(f'Не удалось запустить Vulkan для автозаписи.\n{tail}')
            yield request
        finally:
            with converter.lock:
                if process.poll() is None:
                    process.terminate()
                converter.child = None
            process.wait(timeout=30)


def collect(converter, manifest, reference, recipe, names, directory, detect, *, require_all=True):
    if len(names) > 4096 or any(re.fullmatch(r'[A-Za-z0-9_.-]{1,200}', name) is None for name in names):
        raise ValueError('Имена выбранных тензоров несовместимы с автозаписью.')
    (directory / 'tensors.txt').write_text('\n'.join(names) + '\n', encoding='ascii')
    settings, _, prompt, seed = runtime_config(recipe, manifest, reference)
    payload = render_spec(settings, prompt, seed, detect)
    converter.log('Собираю активации на Vulkan.')
    with worker(converter, settings, directory, activations=True) as request:
        result = request('/sdapi/v1/txt2img', payload)
        if not result.get('images') and not (result.get('animated') and result.get('extra_data')):
            raise RuntimeError('Vulkan не вернул результат калибровочного прогона.')
    captured = {}
    for i, name in enumerate(names):
        path = directory / f'{i}.f32'
        if not path.is_file():
            if not require_all:
                continue
            raise ValueError(f'Во время генерации не найдены активации {name}.')
        with path.open('rb') as data:
            shape = tuple(map(int, np.fromfile(data, dtype='<u4', count=2)))
            if len(shape) != 2 or not 2 <= shape[0] <= 32 or not 32 <= shape[1] <= 65536:
                raise ValueError(f'Некорректный размер активаций {name}.')
            rows = np.fromfile(data, dtype='<f4').reshape(shape)
        if not np.isfinite(rows).all():
            raise ValueError(f'Некорректные значения активаций {name}.')
        output = directory / f'{i}.npy'
        np.save(output, rows, allow_pickle=False)
        captured[name] = output
        converter.log(f'{name} · {shape[0]} активаций')
    return captured, settings, payload


def render(converter, settings, payload, directory):
    directory.mkdir()
    with worker(converter, settings, directory, activations=False) as request:
        return request('/sdapi/v1/txt2img', payload)


def pixels(result):
    from PIL import Image

    animated = bool(result.get('animated'))
    data = None
    options = list((result.get('images') or [])[:1]) + [result.get('extra_data')]
    for encoded in options:
        if not isinstance(encoded, str) or not 0 < len(encoded) <= 32 * 1024 * 1024:
            continue
        decoded = base64.b64decode(encoded.split(',', 1)[-1], validate=True)
        if not animated or (decoded.startswith(b'RIFF') and decoded[8:12] == b'AVI '):
            data = decoded
            break
    if data is None:
        raise ValueError('Неверный ответ Vulkan для оценки рендера.')
    if animated:
        def frames(start, end, in_movie=False):
            while start + 8 <= end:
                kind, size = struct.unpack_from('<4sI', data, start)
                content, stop = start + 8, start + 8 + size
                if stop > end:
                    raise ValueError('Повреждён блок AVI.')
                if kind == b'LIST' and data[content:content+4] in (b'movi', b'rec '):
                    yield from frames(content + 4, stop, True)
                elif in_movie and kind[2:] in (b'dc', b'db'):
                    yield data[content:stop]
                start = stop + (size & 1)

        files = list(frames(12, min(len(data), struct.unpack_from('<I', data, 4)[0] + 8)))
        if len(files) != 5:
            raise ValueError('Ожидаются пять кадров в калибровочном видео.')
    else:
        files = [data]
    rows = []
    for file in files:
        with Image.open(io.BytesIO(file)) as frame:
            if frame.width * frame.height > 2_000_000:
                raise ValueError('Размер калибровочного кадра превышает допустимый.')
            rows.append(np.asarray(frame.convert('RGB'), dtype=np.uint8))
    return np.stack(rows)
