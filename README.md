# GGUF Converter for Windows

[Русский](README.ru.md) · [Release / Скачать](https://github.com/aktalascloud/gguf-converter/releases/tag/v1.9.0) · [Guide / Документация](docs/calibration.md) · [Evidence / Данные](https://github.com/aktalascloud/gguf-converter/blob/28aa603895082e0f42170ecda1dbc69e5acd598a/benchmarks/2026-11-17-whole-gsq-rco-rx580.json)

Convert text and media checkpoints to GGUF in one portable Windows EXE with Python included. Compare sizes before converting and calibrate supported weights on AMD Polaris through Vulkan.

**Quick start.** Run `GGUF-Converter.exe`, open a GGUF, Safetensors file or Hugging Face model folder and choose a target tier. Keep the original checkpoint for calibration when available.

**Evidence.** The November 17, 2026 RX 580 GSQ/RCO run reduced UMT5-XXL projection error from 60,640 to 37,930 on captured activations within the tensor-byte budget. The [published encoder](https://huggingface.co/TalasCloud/UMT5-XXL-Encoder-GSQ-RCO-GGUF) is 3.62 GB. Projection error measures calibration fit.

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

[RX 580 benchmarks](benchmarks/README.md) · [Patches](patches/README.md) · [License and components](NOTICE.md)

[All projects / Все проекты](https://github.com/aktalascloud)
