# GGUF Converter

[Русский](README.ru.md) · [Download for Windows](https://github.com/aktalascloud/gguf-converter/releases/latest) · [CLI manifests and build](docs/calibration.md)

A single Windows EXE for converting text and media models to GGUF. It bundles Python, detects the model type and estimates file size and GPU memory use.

Open a GGUF, Safetensors file or Hugging Face model folder. Choose Q2, Q4, Q6 or Q8. Supported source packing includes scaled INT8, FP8 and selected INT4 schemes with their metadata.

## Calibration

| Switch | Use when |
| --- | --- |
| **Off** | You want ordinary quantization. |
| **Auto** | You want ACTCAL at the current quant tier or GSQ/RCO for a different tier. |
| **ACTCAL** | You want to fit block codes or scales while preserving each tensor type. |
| **GSQ/RCO** | You want to try quant assignments under a size budget. |
| **MACaQ-R** | You want to refine codes while retaining the current GGUF layout and scales. |

Calibration uses Vulkan on supported AMD Polaris devices. Use the matching original checkpoint where available. ACTCAL can also audit an exact F32 reconstruction of the current GGUF and preserve its bytes. A matching KoboldCpp profile can capture photo, video and YuE2 activations. [Calibration guide](docs/calibration.md)

The bundled MACaQ-R adaptation uses 32-column Hessian blocks and the existing inference kernels. [GSQ](https://github.com/IST-DASLab/GSQ) · [RCO](https://github.com/IST-DASLab/RCO) · [MACaQ](https://github.com/huawei-csl/MACaQ)

[Whole-checkpoint RX 580 comparison](benchmarks/2026-11-17-whole-gsq-rco-rx580.json) · [All benchmarks](benchmarks/) · [Patches](patches/) · [License and components](NOTICE.md)
