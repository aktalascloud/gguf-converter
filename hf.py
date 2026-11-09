"""Run the pinned llama.cpp converter and decode the standard AWQ GEMM layout."""

from pathlib import Path
import json
from math import prod
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import BACKENDS

backend = BACKENDS / 'hf'
sys.path[:0] = [str(backend), str(backend / 'gguf-py')]
from conversion.base import LazyTorchTensor, ModelBase
from convert_hf_to_gguf import main


def unpack_awq(packed):
    # AutoAWQ stores eight output channels in GEMM order 0,2,4,6,1,3,5,7.
    reverse_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=packed.device)
    shifts = reverse_order * 4
    return ((packed.unsqueeze(-1) >> shifts) & 15).flatten(1)


def decode_awq(weight, zeros, scales, group_size):
    groups = scales.shape[0]
    unpacked = unpack_awq(weight).reshape(groups, group_size, scales.shape[1]).float()
    offsets = unpack_awq(zeros).unsqueeze(1)
    return ((unpacked - offsets) * scales.float().unsqueeze(1)).flatten(0, 1).T.contiguous()


def adapt_awq(model, config):
    if (config.get('version', '').lower() != 'gemm' or config.get('bits') != 4
            or config.get('zero_point') is not True):
        raise ValueError('Поддерживается AWQ GEMM 4-bit с zero_point=true.')
    tensors = model.model_tensors
    names = [name for name in tensors if name.endswith('.qweight')]
    if not names:
        raise ValueError('Метаданные AWQ указаны, но тензоры qweight отсутствуют.')
    for name in names:
        base = name.removesuffix('.qweight')
        if base + '.weight' in tensors:
            raise ValueError(f'Два представления веса {base}.')
        weight = tensors.pop(name)()
        zeros = tensors.pop(base + '.qzeros')()
        scales = tensors.pop(base + '.scales')()
        if len(weight.shape) != 2 or weight.dtype != torch.int32 or zeros.dtype != torch.int32:
            raise ValueError(f'Неверный формат упаковки AWQ {base}.')
        in_features, packed_columns = weight.shape
        group = config.get('group_size')
        if group == -1:
            group = in_features
        if type(group) is not int or group <= 0 or in_features % group:
            raise ValueError(f'Недопустимый размер группы AWQ {base}.')
        if (zeros.shape != (in_features // group, packed_columns)
                or scales.shape != (in_features // group, packed_columns * 8)
                or not scales.dtype.is_floating_point):
            raise ValueError(f'Масштабы и zero points не совпадают с формой AWQ {base}.')
        meta = LazyTorchTensor.meta_with_dtype_and_shape(torch.float32, (packed_columns * 8, in_features))
        tensors[base + '.weight'] = lambda meta=meta, args=(weight, zeros, scales, group): LazyTorchTensor(
            meta=meta, args=args, func=decode_awq)


def decode_bnb4(packed, absmax, codebook, shape, block, nested):
    count = prod(shape)
    packed = packed.flatten()
    absmax = absmax.flatten()
    if packed.dtype != torch.uint8 or packed.numel() != (count + 1) // 2 or codebook.numel() != 16:
        raise ValueError('Недопустимая упаковка bitsandbytes 4-bit.')
    if absmax.numel() != (count + block - 1) // block:
        raise ValueError('Число блоков bitsandbytes не совпадает с формой веса.')
    if nested is not None:
        nested_max, nested_map, nested_block, offset = nested
        if (absmax.dtype != torch.uint8 or nested_map.numel() != 256
                or nested_max.numel() != (absmax.numel() + nested_block - 1) // nested_block):
            raise ValueError('Повреждены масштабы двойного квантования bitsandbytes.')
        decoded = nested_map.flatten().float()[absmax.to(torch.int32)]
        block_scales = nested_max.flatten().float().repeat_interleave(nested_block)
        absmax = decoded * block_scales[:absmax.numel()] + offset
    elif not absmax.dtype.is_floating_point:
        raise ValueError('Для целочисленных масштабов bitsandbytes нужна вложенная схема.')
    codes = torch.empty(packed.numel() * 2, dtype=torch.int32)
    codes[::2] = packed >> 4
    codes[1::2] = packed & 15
    decoded = codebook.flatten().float()[codes[:count]]
    scales = absmax.float().repeat_interleave(block)[:count]
    return (decoded * scales).reshape(shape)


def adapt_bnb(model):
    tensors = model.model_tensors
    for name in tuple(tensors):
        marker = '.quant_state.bitsandbytes__'
        if marker in name:
            weight_name, kind = name.split(marker, 1)
            state = json.loads(bytes(LazyTorchTensor.to_eager(tensors.pop(name)()).flatten().tolist()))
            shape = state['shape']
            block = state['blocksize']
            if (kind not in {'nf4', 'fp4'} or state.get('quant_type') != kind or not shape
                    or any(type(n) is not int or n <= 0 for n in shape) or type(block) is not int or block <= 0):
                raise ValueError(f'Неизвестное состояние bitsandbytes {name}.')
            args = [tensors[weight_name](), tensors.pop(weight_name + '.absmax')(), tensors.pop(weight_name + '.quant_map')()]
            nested = None
            if 'nested_blocksize' in state:
                nested_block = state['nested_blocksize']
                if type(nested_block) is not int or nested_block <= 0:
                    raise ValueError(f'Неизвестная вложенная схема bitsandbytes {name}.')
                nested = (tensors.pop(weight_name + '.nested_absmax')(), tensors.pop(weight_name + '.nested_quant_map')(),
                          nested_block, float(state['nested_offset']))
            meta = LazyTorchTensor.meta_with_dtype_and_shape(torch.float32, shape)
            tensors[weight_name] = lambda meta=meta, args=(*args, shape, block, nested): LazyTorchTensor(
                meta=meta, args=args, func=decode_bnb4)
        elif name.endswith('.SCB'):
            base = name.removesuffix('.SCB')
            weight_name = base + '.weight'
            if (layout := tensors.pop(base + '.weight_format', None)) is not None:
                if LazyTorchTensor.to_eager(layout()).item() != 0:
                    raise ValueError(f'Поддерживается построчная упаковка bitsandbytes INT8: {base}.')
            weight, scales = tensors[weight_name](), tensors.pop(name)()
            if (weight.dtype != torch.int8 or len(weight.shape) != 2 or scales.shape != (weight.shape[0],)
                    or not scales.dtype.is_floating_point):
                raise ValueError(f'Масштабы SCB не совпадают с формой INT8 {base}.')
            tensors[weight_name] = lambda w=weight, s=scales: w.float() * (s.float().unsqueeze(1) / 127)


original_dequant = ModelBase.dequant_model


def dequant_checked(model):
    config = model.hparams.get('quantization_config') or {}
    if config.get('quant_method') == 'awq':
        adapt_awq(model, config)
    elif config.get('quant_method') == 'bitsandbytes':
        adapt_bnb(model)
    else:
        original_dequant(model)
    for name, get_tensor in model.model_tensors.items():
        if name.endswith(('.qweight', '.qzeros', '.weight_packed')) or '.quant_state.' in name:
            raise ValueError(f'Остался недекодированный упакованный тензор {name}.')
        if name.endswith('.weight') and not get_tensor().dtype.is_floating_point:
            raise ValueError(f'Для целочисленного веса {name} не найден поддерживаемый декодер.')


if __name__ == '__main__':
    ModelBase.dequant_model = dequant_checked
    main()
