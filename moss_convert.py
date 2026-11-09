"""Decode Artius media checkpoints, adapted from KoboldCpp Media's MossConvert.py."""

from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import comfy_formats, gguf_module, safe_header


def unrotate(values, size=256):
    if (type(size) is not int or size < 4 or size & (size - 1)
            or (size.bit_length() - 1) % 2 or values.shape[-1] % size):
        raise ValueError('Некорректная геометрия ConvRot.')
    result = np.array(values, dtype=np.float32, copy=True).reshape(-1, size)
    stride = 1
    while stride < size:
        block = result.reshape(-1, size // (stride * 4), 4, stride)
        a, b, c, d = [block[:, :, index, :].copy() for index in range(4)]
        total = a + b + c + d
        for index, plane in enumerate((d, c, b, a)):
            block[:, :, index, :] = total - 2 * plane
        stride *= 4
    return (result / np.sqrt(np.float32(size))).reshape(values.shape)


def fp8_table():
    bits = np.arange(256, dtype=np.uint16)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    value = np.where(exponent == 0, mantissa / 512,
                     (1 + mantissa / 8) * np.exp2(exponent.astype(np.int16) - 7))
    value = np.where((exponent == 15) & (mantissa == 7), np.nan, value)
    return (value * np.where(bits & 128, -1, 1)).astype(np.float32)


FP8 = fp8_table()
DTYPES = {'F32': '<f4', 'F16': '<f2', 'BF16': '<u2', 'F8_E4M3': 'u1',
          'I8': 'i1', 'U8': 'u1', 'I32': '<i4', 'I64': '<i8'}


class SafeFile:
    def __init__(self, path):
        self.header, metadata, self.offset = safe_header(path)
        self.formats = comfy_formats(path, self.header, metadata, self.offset)
        self.data = np.memmap(path, mode='r', dtype=np.uint8)

    def array(self, name, rows=None):
        tensor = self.header[name]
        start, end = tensor['data_offsets']
        kind = tensor['dtype']
        data = self.data[self.offset + start:self.offset + end].view(DTYPES[kind]).reshape(tensor['shape'])
        if rows is not None:
            data = data[rows]
        if kind == 'BF16':
            return (data.astype(np.uint32) << 16).view(np.float32)
        if kind == 'F8_E4M3':
            return FP8[data]
        return data


def decode_rows(source, name, rows, config):
    base = name.removesuffix('.weight')
    scheme = config.get('format', '')
    packed = scheme == 'asym_w4a8_int8'
    raw = source.array(name, rows)
    if packed:
        raw = raw.view(np.uint8)
        codes = np.empty((raw.shape[0], raw.shape[1] * 2), np.uint8)
        codes[:, 0::2] = raw & 15
        codes[:, 1::2] = raw >> 4
        codebook = source.array(base + '.weight_codebook').astype(np.float32)
        relative = source.array(base + '.weight_s_rel', rows)
        levels = codebook[codes].reshape(raw.shape[0], -1, config['group_size'])
        values = np.clip(np.rint(levels * relative[:, :, None]), -127, 127).reshape(codes.shape)
    else:
        values = raw.astype(np.float32)
    if config.get('convrot'):
        values = unrotate(values, config.get('convrot_groupsize', 256))
    if packed:
        values *= source.array(base + '.weight_s_channel', rows).reshape(-1, 1)
    else:
        scale_key = base + '.weight_scale'
        if scale_key in source.header:
            scale = source.array(scale_key)
            values *= scale.reshape(-1, 1)[rows] if scale.size > 1 else scale.item()
    if not np.isfinite(values).all():
        raise ValueError(f'Декодирование дало Inf/NaN в {name}.')
    return values


def convert(source, output, role, quantization):
    constants = gguf_module('constants')
    kind = constants.GGMLQuantizationType[quantization]
    block_size, block_bytes = constants.GGML_QUANT_SIZES[kind]
    quantize = gguf_module('quants').quantize
    writer_class = gguf_module('gguf_writer').GGUFWriter
    data = SafeFile(source)
    tempfile.tempdir = str(output.parent)
    writer = writer_class(output, 'moss_image' if role == 'image' else 'qwen35', use_temp_file=True)
    writer.add_name(source.stem)
    writer.add_file_type(getattr(constants.LlamaFileType, 'MOSTLY_' + quantization))
    auxiliary = ('.comfy_quant', '.weight_scale', '.scale_weight', '.weight_s_rel',
                 '.weight_s_channel', '.weight_codebook', '.weight_correction')
    scratch = output.with_name('tensor.bin')
    try:
        for name, tensor in data.header.items():
            if name.endswith(auxiliary):
                continue
            if role == 'image':
                mapped = 'model.diffusion_model.' + name.removeprefix('model.diffusion_model.')
            else:
                mapped = name.replace('model.language_model.', 'model.', 1)
            config = data.formats.get(name.removesuffix('.weight'), {})
            shape = tensor['shape']
            if config.get('format') == 'asym_w4a8_int8':
                shape = [shape[0], shape[1] * 2]
            if len(shape) != 2 or shape[1] % block_size or not name.endswith('.weight'):
                raw = np.array(data.array(name), copy=True)
                if raw.dtype not in (np.float32, np.float16, np.int32):
                    raw = raw.astype(np.int32 if raw.dtype == np.int64 else np.float32)
                writer.add_tensor(mapped, raw)
                continue
            rows, columns = shape
            encoded = np.memmap(scratch, mode='w+', dtype=np.uint8,
                               shape=(rows, columns // block_size * block_bytes))
            try:
                for start in range(0, rows, 256):
                    part = slice(start, min(rows, start + 256))
                    encoded[part] = quantize(decode_rows(data, name, part, config), kind)
                writer.add_tensor(mapped, encoded, raw_dtype=kind)
            finally:
                encoded._mmap.close()
                scratch.unlink()
            print(f'{name} → {quantization}', flush=True)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
        data.data._mmap.close()


if __name__ == '__main__':
    convert(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4])
