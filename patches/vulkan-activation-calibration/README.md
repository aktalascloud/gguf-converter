# Preserve Q4_0/Q8_0 while calibrating every observed matrix

Adds `ACTCAL-Q4` and `ACTCAL-Q8`, a blockwise Vulkan solver that fits scales and codes against activation-weighted errors. The calibration preserves each tensor's layout and accepts candidate rows based on a validation projection. Capture supports media tensors with contiguous F16 or F32 activations and reports coverage. The existing GSQ/RCO manifests still work independently.

Base `bb96e56438adb325c07909c1d486e6995fea048b`. Applies after `safetensors-fp8-e5m2`. See the [RX 580 release checks](https://github.com/aktalascloud/gguf-converter/blob/b960e0cee060e05e135fdaab07632f7710dcbba4/benchmarks/2026-11-16-v160-rx580.json).
