# GGUF Converter for Windows

[Русский](README.ru.md) · [Download the portable EXE](https://github.com/aktalascloud/gguf-converter/releases/latest)

Convert local model weights to **Q2, Q4, Q6 or Q8** on Windows 10/11 x64. Open `GGUF-Converter.exe`, select a supported file or Hugging Face directory, and choose the quantization. The window detects the model and hardware, estimates the resulting file size before conversion, and shows the active stage and exact size after conversion. Python and the conversion tools are bundled.

| Input | Conversion path |
|---|---|
| Text GGUF | llama.cpp, including sharded models and supported requantization |
| Local Hugging Face folder | Supported architectures with config, tokenizer and weights. FP32, FP16, BF16, supported FP8, GPTQ, AWQ GEMM 4-bit with zero points, supported compressed-tensors layouts and bitsandbytes NF4/FP4 or row-wise INT8 |
| Diffusion weights | GGUF, Safetensors and CKPT through stable-diffusion.cpp. Includes scaled FP8 E4M3/E5M2 and Comfy INT8 tensorwise Safetensors. Wan architecture metadata is preserved or detected from the tensor names. |
| Packed ComfyUI weights | Supported tensorwise INT8, scaled FP8, and W4A8 layouts with their metadata |

The converter writes a separate GGUF and protects existing files from replacement. Tensors that cannot use the chosen block layout retain a compatible representation. New or proprietary architectures and packing schemes require a matching loader. Hugging Face K-quant conversion may stage an FP32 GGUF. Some scaled diffusion checkpoints also need temporary FP32 space.

### Target hardware

CPU instructions and GPU memory are detected for the information panel and memory estimates. Q4 defaults to `Q4_K` in the window. Choose `--quantization Q4_0` in the CLI for that exact layout. The inference engine applies AVX2 and Vulkan kernels. An F16 KV-cache estimate is available for recognized text architectures. Separate component weights and inference buffers add to memory use. [GGML runtime repacking](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h) · [Vulkan memory budget](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_memory_budget.html)

## Vulkan GSQ-RCO preview

Enable **Weight calibration** and open a media GGUF or a `gguf-gsq-rco-vulkan-v1` manifest. The bundled calibration uses Vulkan shaders on a supported AMD Polaris GPU. The media worker needs AVX2 to record activations. A matching local KoboldCpp profile supplies companion weights automatically, including when the selected GGUF is its text encoder. Mixed-layout GSQ/RCO from a directly selected GGUF calibrates two recognized attention projections. A manifest can cover up to eight matrices and supports Q3 through the CLI. [GSQ](https://github.com/IST-DASLab/GSQ) · [RCO](https://github.com/IST-DASLab/RCO)

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

Earlier local-batch calibration evidence is recorded by binary SHA. [v1.2.0 RX 580 measurements](benchmarks/2026-11-16-rx580.json)

The v1.3.0 RX 580 check covers automatic capture on ABDM and Wan, photo generation without upscaling, a 480p five-frame clip, and GGUF sizes for all four standard formats. [v1.3.0 results](benchmarks/2026-11-16-v130-rx580.json)

### Calibration with the original tensor layouts

For a model dominated by Q4_0 or Q8_0, enable weight calibration and select the existing tier. Provide the matching GGUF or Safetensors checkpoint. `ACTCAL-Q4` and `ACTCAL-Q8` use activation-weighted least squares on Vulkan. Each candidate matrix is measured on 8–32 captured rows split into fitting and selection sets. The converter keeps tensor types, model metadata, and payload size. Check output quality separately on independent prompts.

For the CLI, use `-q ACTCAL-Q4` or `-q ACTCAL-Q8` with a `gguf-activation-calibration-v1` manifest. Set `base`, `reference` and `capture` the same way as for GSQ/RCO. Under `targets`, use `"4": {"all": true}` or `"8": {"all": true}` to capture and fit every reachable Q4_0 and Q8_0 matrix. Use `tensors` with names mapped to `.npy` or `null` for explicit selections. Safetensors inputs support common floating formats, scaled INT8, and W4A8 INT4 with its codebook and scales. Same-tier K-quant GSQ uses a GGUF reference and the `GSQ-RCO` modes.

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

To include the Vulkan shaders and helper, pass all four options `--cxx`, `--glslc`, `--vulkan-include` and `--vulkan-import`. Point them to the MinGW-w64 `g++.exe`, Vulkan SDK `glslc.exe`, SDK include directory and `vulkan-1.lib`. Pass `--native-capture` with the directory produced by `vulkan/build_capture.py` to enable the embedded media worker. [Corresponding source and notices](NOTICE.md)

[License](LICENSE.md) · [Corresponding sources and component notices](NOTICE.md).
