# Calibration and build

## GSQ/RCO on Vulkan

Choose **GSQ/RCO** under Calibration and open a media GGUF or a `gguf-gsq-rco-vulkan-v1` manifest. The bundled calibration uses Vulkan shaders on a supported AMD Polaris GPU. The media worker needs AVX2 to record activations. A matching local KoboldCpp profile supplies companion weights automatically, including when the selected GGUF is its text encoder. Mixed-layout GSQ/RCO from a directly selected GGUF calibrates two recognized attention projections. A manifest can cover up to eight matrices and supports Q3 through the CLI. [GSQ](https://github.com/IST-DASLab/GSQ) · [RCO](https://github.com/IST-DASLab/RCO)

```json
{
  "format": "gguf-gsq-rco-vulkan-v1",
  "base": "model.gguf",
  "targets": {
    "3": {
      "max_tensor_bytes": 1000000000,
      "tensors": {
        "blk.0.attn_q.weight": "attention-input.npy",
        "blk.0.ffn_up.weight": "mlp-input.npy"
      }
    }
  }
}
```

Paths are relative to the manifest. Use a NumPy `.npy` array of 2–32 activation rows for a matrix or `null` for capture. For automatic capture add `"capture": {"prompt": "…", "profile": "photo.kcpps"}`. A profile is optional for standalone diffusion models. Use `capture.base_part` to replace an encoder such as `sdclip1` and `capture.profile_root` to resolve relative paths in the profile. Enable `"model_loss": true` to rerank the feasible GSQ/RCO and rounding candidates by pixel MSE on a complete calibration render. The measured loss covers the complete output. GSQ still learns per-matrix weights, and RCO proposes a bounded set of configurations under an exact tensor-byte budget.

Earlier local-batch calibration evidence is recorded by binary SHA. [v1.2.0 RX 580 measurements](../benchmarks/2026-11-16-rx580.json)

The v1.3.0 RX 580 check covers automatic capture on ABDM and Wan, photo generation without upscaling, a 480p five-frame clip, and GGUF sizes for all four standard formats. [v1.3.0 results](../benchmarks/2026-11-16-v130-rx580.json)

### Calibration with the original tensor layouts

Enable weight calibration, select the existing tier and provide the matching GGUF or Safetensors checkpoint. ACTCAL uses activation-weighted least squares on Vulkan. For Q4_0 and Q8_0 it fits codes and scales. For Q2_K through Q6_K it fits superblock scales with the existing packed codes and subscales. Candidate rows are selected by projection error on a separate subset of the recorded activations. Tensor types, model metadata and payload size are preserved.

CLI ACTCAL tiers are Q2, Q3, Q4, Q5, Q6 and Q8. Use `gguf-activation-calibration-v1` with `base`, `reference`, and `capture`. For a mixed Q4_K_M encoder, set `"4": {"all": true}` under `targets` to preserve each matrix's Q4_K or Q6_K type. For explicit selection, set `tensors` to names mapped to `.npy` files or `null`. Safetensors references support common floating formats, scaled INT8 and W4A8 INT4 with scales and codebooks. The earlier GSQ/RCO workflow has its own manifest.

For YuE2, the converter detects the music GGUF and looks for a matching KoboldCpp music profile. A manifest can set `capture.base_part` to `musicllm` and provide `musicvae` in `capture.models`. `capture.prompt` is the musical style, and `capture.lyrics` optionally supplies training lyrics. The capture worker records Vulkan inputs to AR and NAR matrix products. Lookup tables and tensors unused by the calibration request keep their original bytes.

When only a quantized GGUF is available, select its current tier, choose **ACTCAL** and select **No** in the reference checkpoint dialog. The CLI equivalent is `"reference": null`. The round trip expands weights to F32 in bounded chunks, runs the requantizer and checks activation projections. The original GGUF already has zero reconstruction error against those F32 values, so the published output retains its exact bytes.

Choose **MACaQ-R** under Calibration to fit stored codes with fixed scales using a Vulkan adaptation of the greedy refinement. CLI manifests set `"refiner": "macaq-r"` inside the target. Supported layouts are Q4_0, Q8_0, Q4_K, Q5_K and Q6_K. The adaptation uses 32-column Hessian blocks and accepts rows on the held-out activation subset. These GGUFs use the existing inference kernels and activation precision. [MACaQ paper](https://arxiv.org/abs/2611.10649) · [MACaQ-R guide](https://github.com/huawei-csl/MACaQ/blob/3810efca0b9e819c8eb51addd93c3c87225b7476/llamacpp/README.md#macaq-r)

The RX 580 trials covered 168 UMT5 projections and 395 YuE2 matrices. The stock and calibrated generation time ranges overlapped in the fixed-profile repetitions. The recorded visual and vocal content checks are included with the timings. [UMT5 and YuE2 measurements](../benchmarks/2026-11-17-umt5-yue2-rx580.json)

## GSQ-RCO assembly

Select **GSQ-RCO-Q2**, **GSQ-RCO-Q3** or **GSQ-RCO-Q4** and open a JSON manifest. The converter copies packed tensors from previously GSQ-optimized GGUF variants according to a per-tensor RCO assignment. Run GSQ calibration and the RCO search separately. The bundled EXE handles the assembly offline. [GSQ](https://github.com/IST-DASLab/GSQ) · [RCO](https://github.com/IST-DASLab/RCO)

```json
{
  "format": "gguf-gsq-rco-v1",
  "base": "model-base.gguf",
  "variants": {
    "q2": "model-gsq-q2.gguf",
    "q3": "model-gsq-q3.gguf"
  },
  "targets": {
    "2": {
      "tensors": {
        "blk.0.attn_q.weight": "q2",
        "blk.0.ffn_up.weight": "q3"
      }
    }
  }
}
```

Paths are relative to the manifest. Add one entry under each target you want to assemble. Each assignment value names a file in the variants map. Tensors outside the assignment come from the base file. The optional max_tensor_bytes applies to the total tensor payload, including tensors kept from the base. The converter prints the resulting bits per parameter.

To run a conversion from PowerShell, pass `--cli` and wait for the process to exit.

```powershell
Start-Process -FilePath '.\GGUF-Converter.exe' -ArgumentList '--cli "D:\Models\My model" -q Q4_0 -t 8' -Wait -PassThru
```

For independent targets use `--gpu vulkan --cpu avx`, or `--gpu pascal --cpu avx2`. Use `-q Q4` for target-aware selection. An explicit `-q Q4_K` or `-q Q4_0` fixes the tensor format.

To build from source, run the following on Windows with Python 3.12+ and the Windows .NET Framework C# compiler available. The build uses the pinned checksums in `dependencies.lock.json` and writes the EXE to `dist/`.

```powershell
py -3.12 -I -B portable_build.py
```

To include the Vulkan shaders and helper, pass all four options `--cxx`, `--glslc`, `--vulkan-include` and `--vulkan-import`. Point them to the MinGW-w64 `g++.exe`, Vulkan SDK `glslc.exe`, SDK include directory and `vulkan-1.lib`. Pass `--native-capture` with the directory produced by `vulkan/build_capture.py` to enable the embedded media worker. [Corresponding source and notices](../NOTICE.md)

[License](../LICENSE.md) · [Corresponding sources and component notices](../NOTICE.md).
