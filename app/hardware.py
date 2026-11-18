"""Observe host hardware and estimate a GGUF memory budget."""
from dataclasses import dataclass
import ctypes
import json
import os
import subprocess
from gpu_adapter import select_dml_adapter


GIB = 1024**3


@dataclass(frozen=True)
class Hardware:
    instructions: tuple[str, ...] = ()
    gpu: str = ''
    vram_bytes: int = 0
    budget_bytes: int | None = None
    polaris: bool = False
    gsq: bool = False

    @property
    def gpu_target(self):
        return 'polaris' if self.polaris else 'auto'

    @property
    def cpu_target(self):
        return 'avx2' if 'AVX2' in self.instructions else 'avx' if 'AVX' in self.instructions else 'auto'


def detect(helper):
    if os.name != 'nt':
        return Hardware()
    instruction_ids = (('SSE4.2', 38), ('AVX', 39), ('AVX2', 40), ('AVX-512F', 41))
    has_feature = ctypes.WinDLL('kernel32', use_last_error=True).IsProcessorFeaturePresent
    has_feature.argtypes = [ctypes.c_uint]
    has_feature.restype = ctypes.c_int
    instructions = tuple(name for name, number in instruction_ids if has_feature(number))
    try:
        result = subprocess.run([str(helper), '--probe'], capture_output=True, encoding='utf-8',
                                errors='replace', check=True, timeout=8,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        rows = []
        for line in result.stdout.splitlines():
            vendor, device, vram, budget, available, gsq, name = line.split('\t', 6)
            polaris = int(vendor) == 0x1002 and int(device) in (0x67df, 0x6fdf)
            rows.append(Hardware(instructions, json.loads(name), int(vram),
                                 int(available) if int(budget) else None, polaris, bool(int(gsq))))
        if rows:
            return max(rows, key=lambda item: item.vram_bytes)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        adapter = select_dml_adapter()
        polaris = adapter['vendor'] == 0x1002 and adapter['device'] in (0x67df, 0x6fdf)
        return Hardware(instructions, adapter['name'], adapter['dedicated_bytes'], polaris=polaris)
    except (RuntimeError, OSError):
        return Hardware(instructions)


def gguf_sizes(reader, layouts):
    """Estimate complete files from tensor geometry and GGUF alignment."""
    result = {}
    alignment = reader.alignment
    aligned = lambda length: (length + alignment - 1) // alignment * alignment
    for label, (block, block_bytes) in layouts.items():
        total = reader.data_offset
        for tensor in reader.tensors:
            size = tensor.n_bytes
            if len(tensor.shape) == 2 and tensor.name.endswith('.weight') and int(tensor.shape[0]) % block == 0:
                size = int(tensor.n_elements) // block * block_bytes
            total += aligned(size)
        result[label] = total
    return result


def safetensor_sizes(tensors, layouts):
    """Rough payload estimate for unpacked floating-point weights."""
    scalar_bytes = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2}
    parameters = []
    for name, info in tensors.items():
        shape = info['shape']
        dtype = info['dtype']
        if name.endswith('.weight') and len(shape) == 2:
            if dtype not in scalar_bytes:
                return {}
            parameters.append((shape[0], shape[1], scalar_bytes[dtype]))
        elif any(marker in name for marker in ('.comfy_quant', '.quant_state.', '.weight_scale', '.qweight')):
            return {}
    if not parameters:
        return {}
    preserved = sum(info['data_offsets'][1] - info['data_offsets'][0] for info in tensors.values())
    return {tier: preserved + sum(rows * (width // block * size - width * raw)
                                  for rows, width, raw in parameters if width % block == 0)
            for tier, (block, size) in layouts.items()}


def kv_geometry(reader):
    """F16 KV bytes per token for recognized uniform-attention decoder models."""
    def field(name):
        item = reader.get_field(name)
        return item.contents() if item is not None else None
    arch = field('general.architecture')
    if arch not in ('llama', 'qwen2', 'qwen3') or not field('tokenizer.ggml.model'):
        return None
    def positive(key):
        value = field(f'{arch}.{key}')
        return value if type(value) is int and value > 0 else 0
    layers, context = positive('block_count'), positive('context_length')
    heads, kv_heads = positive('attention.head_count'), positive('attention.head_count_kv')
    width = positive('embedding_length')
    if not layers or not context or not heads or width % heads:
        return None
    k = positive('attention.key_length') or width//heads
    v = positive('attention.value_length') or width//heads
    if not k or not v:
        return None
    return 2 * layers * (kv_heads or heads) * (k + v), context


def advice(sizes, card, *, llm=False, geometry=None):
    if not sizes or not card.vram_bytes:
        return None, None
    budget = min(card.vram_bytes, card.budget_bytes) if card.budget_bytes is not None else card.vram_bytes
    reserve = max(GIB, budget // 5) if llm else max(2 * GIB, budget // 4)
    usable = max(0, budget - reserve)
    if llm and geometry is None:
        return None, None
    if geometry:
        bytes_per_token, maximum = geometry
        smallest_context = min(2048, maximum)
        picked = next((tier for tier, size in reversed(sizes.items())
                       if size + bytes_per_token * smallest_context <= usable), None)
        if picked is None:
            return None, None
        turns = min(maximum, (usable - sizes[picked]) // bytes_per_token)
        return picked, turns if turns < 256 else turns // 256 * 256
    return next((tier for tier, size in reversed(sizes.items()) if size <= usable), None), None
