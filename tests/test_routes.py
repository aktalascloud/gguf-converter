"""Smoke-test dispatch when GPU and CPU targets are selected independently."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert import media_role, output_quantization


assert media_role('wan2.2', ()) == 'Видео'
assert media_role('QwenImageTransformer2DModel', ()) == 'Фото'
assert media_role('qwen3_5', ()) == ''
assert output_quantization('Q4') == 'Q4_K'
assert output_quantization('Q4', 'vulkan', 'avx') == 'Q4_0'
assert output_quantization('Q4', 'pascal', 'avx2') == 'Q4_0'
assert output_quantization('Q4_K', 'pascal', 'avx') == 'Q4_K'
assert output_quantization('Q6', 'polaris', 'avx2') == 'Q6_K'
assert output_quantization('Q8', 'cuda', 'avx') == 'Q8_0'
try:
    output_quantization('GSQ-RCO-Q8', 'cuda', 'avx')
except ValueError:
    pass
else:
    raise AssertionError('Hardware presets must not modify a GSQ/RCO plan.')
