"""Expand media weights whose scales are stored separately in Safetensors."""

import json
from math import prod
from pathlib import Path
import shutil
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from convert import comfy_formats, safe_header
from moss_convert import SafeFile, np, unrotate


def decode_e5m2(raw):
    exponent = ((raw >> 2) & 31).astype(np.int16)
    fraction = (raw & 3).astype(np.float32) / 4
    value = np.ldexp(np.where(exponent == 0, fraction, 1 + fraction), np.maximum(exponent, 1) - 15)
    value = np.where(exponent == 31, np.where(fraction == 0, np.inf, np.nan), value)
    return np.where(raw & 128, -value, value)


def expand(source, output):
    header, metadata, data_start = safe_header(source)
    formats = comfy_formats(source, header, metadata, data_start)
    weights = {}
    consumed = set()
    for name, info in header.items():
        if not name.endswith('.weight'):
            continue
        base = name.removesuffix('.weight')
        scale = base + '.weight_scale'
        kind = info['dtype']
        cfg = formats.get(base, {})
        scheme = cfg.get('format', '')
        if scheme not in {'', 'int8_tensorwise', 'float8_e4m3fn', 'float8_e5m2'}:
            raise ValueError(f'Схема {scheme} у {name} требует отдельного декодера.')
        if kind not in {'I8', 'F8_E4M3', 'F8_E5M2'} and scheme:
            raise ValueError(f'Схема {scheme} не совпадает с типом {kind} у {name}.')
        if kind not in {'I8', 'F8_E4M3', 'F8_E5M2'}:
            continue
        if kind == 'I8' and (scheme != 'int8_tensorwise' or scale not in header):
            raise ValueError(f'Для INT8 {name} нужны int8_tensorwise и weight_scale.')
        if scheme and scale not in header:
            raise ValueError(f'Не найден масштаб {scale}.')
        if base + '.weight_scale_inv' in header or base + '.scale_weight' in header:
            raise ValueError(f'У {name} нестандартный масштаб. Нужна схема блоков исходной модели.')
        if not info['shape'] or not prod(info['shape']):
            raise ValueError(f'Пустой квантованный вес {name}.')
        weights[name] = (scale if scale in header else None, cfg)
        consumed.update(base + suffix for suffix in ('.weight_scale', '.comfy_quant', '.input_scale', '.activation_scale'))

    remaining_schemes = set(formats) - {n.removesuffix('.weight') for n in weights}
    if remaining_schemes:
        raise ValueError(f'Не найдены веса для схемы {min(remaining_schemes)}.')
    layout = {}
    offset = 0
    for name, info in header.items():
        if name in consumed:
            continue
        a, b = info['data_offsets']
        size = prod(info['shape']) * 4 if name in weights else b - a
        layout[name] = dict(dtype='F32' if name in weights else info['dtype'],
                            shape=info['shape'], data_offsets=[offset, offset + size])
        offset += size
    metadata.pop('_quantization_metadata', None)
    if metadata:
        layout['__metadata__'] = metadata
    encoded = json.dumps(layout, separators=(',', ':')).encode('utf-8')
    encoded += b' ' * (-len(encoded) % 8)
    if shutil.disk_usage(output.parent).free < offset + len(encoded) + 8:
        raise OSError(f'Для промежуточных весов нужно {offset / 2**30:.2f} ГиБ свободного места.')
    source_data = SafeFile(source)
    try:
        with output.open('xb') as dst:
            dst.write(struct.pack('<Q', len(encoded)))
            dst.write(encoded)
            for name in layout:
                if name == '__metadata__':
                    continue
                info = header[name]
                a, b = info['data_offsets']
                if name not in weights:
                    for pos in range(a, b, 8 * 1024 * 1024):
                        dst.write(source_data.data[data_start + pos:data_start + min(pos + 8 * 1024 * 1024, b)])
                    continue
                scale_name, cfg = weights[name]
                shape = info['shape']
                scale = np.float32(1)
                if scale_name:
                    scale_info = header[scale_name]
                    if scale_info['dtype'] not in {'F32', 'F16', 'BF16'}:
                        raise ValueError(f'Неверный тип масштаба {scale_name}.')
                    scale = np.asarray(source_data.array(scale_name), dtype=np.float32)
                    if scale.size == 1:
                        scale = scale.reshape(())
                    elif scale.shape and scale.shape[0] == shape[0] and all(n == 1 for n in scale.shape[1:]):
                        scale = scale.reshape((shape[0],) + (1,) * (len(shape) - 1))
                    else:
                        raise ValueError(f'Нужен общий или поканальный масштаб у {name}.')
                rows_per_chunk = max(1, (2 * 1024 * 1024) // max(1, prod(shape[1:])))
                for begin in range(0, shape[0], rows_per_chunk):
                    rows = slice(begin, begin + rows_per_chunk)
                    if info['dtype'] == 'F8_E5M2':
                        raw = source_data.data[data_start + a:data_start + b].reshape(shape)
                        value = decode_e5m2(raw[rows])
                        del raw
                    else:
                        value = source_data.array(name, rows).astype(np.float32)
                    if cfg.get('convrot'):
                        size = cfg.get('convrot_groupsize', 256)
                        if type(size) is not int or size < 4 or size & (size - 1) or (size.bit_length() - 1) % 2 or shape[-1] % size:
                            raise ValueError(f'Неверный размер ConvRot для {name}.')
                        value = unrotate(value, size)
                    value *= scale[rows] if np.ndim(scale) else scale
                    if not np.isfinite(value).all():
                        raise ValueError(f'Масштабирование дало Inf/NaN в {name}.')
                    dst.write(value.astype('<f4', copy=False).tobytes())
                print(f'Декодирован {name}', flush=True)
    finally:
        source_data.data._mmap.close()


if __name__ == '__main__':
    expand(Path(sys.argv[1]), Path(sys.argv[2]))
