# GGUF Converter for Windows

[Русский](README.ru.md) · [Download the portable EXE](https://github.com/aktalascloud/gguf-converter/releases/latest)

Convert local model weights to **Q2, Q4, Q6 or Q8** on Windows 10/11 x64. Open `GGUF-Converter.exe`, select a supported file or Hugging Face directory, and choose the quantization. The app detects text, image and video models from metadata and recognized tensor structures. The window displays the resulting GGUF type and reports the exact file size after conversion. Python and the conversion tools are bundled.

| Input | Conversion path |
|---|---|
| Text GGUF | llama.cpp, including sharded models and supported requantization |
| Local Hugging Face folder | Supported architectures with config, tokenizer and weights. FP32, FP16, BF16, supported FP8, GPTQ, AWQ GEMM 4-bit with zero points, supported compressed-tensors layouts and bitsandbytes NF4/FP4 or row-wise INT8 |
| Diffusion weights | GGUF, Safetensors and CKPT through stable-diffusion.cpp. Includes scaled FP8 E4M3/E5M2 and Comfy INT8 tensorwise Safetensors. Wan architecture metadata is preserved or detected from the tensor names. |
| Packed ComfyUI weights | Supported tensorwise INT8, scaled FP8, and W4A8 layouts with their metadata |

The converter writes a separate GGUF and protects existing files from replacement. Tensors that cannot use the chosen block layout retain a compatible representation. New or proprietary architectures and packing schemes require a matching loader. Hugging Face K-quant conversion may stage an FP32 GGUF. Some scaled diffusion checkpoints also need temporary FP32 space.

### Target hardware

Choose the GPU and CPU targets independently. For example, Vulkan with AVX and CUDA/Pascal P40 with AVX2 both produce standard GGUF. Universal uses `Q2_K`, `Q4_K`, `Q6_K`, or `Q8_0`. Q4 uses `Q4_0` when the GPU is Polaris/Pascal or the CPU target is AVX/AVX2. The other choices retain their standard formats. NVIDIA CUDA / INT8 selects Q8 in the window. The inference runtime selects Vulkan, AVX, or CUDA kernels at load time. CUDA is unnecessary for conversion. Eligible CPU builds can repack `Q4_0` at load time. Pascal P40 supports INT8 dot products with a compatible CUDA runtime. [GGML runtime repacking](https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt) · [NVIDIA Pascal DP4A](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/)

## Vulkan GSQ-RCO preview

Enable **GSQ/RCO (Vulkan)** and open a media GGUF or a `gguf-gsq-rco-vulkan-v1` manifest. A media GGUF uses automatic activation capture in the bundled Vulkan media worker. Provide a calibration prompt and optionally a `.kcpps` profile for the companion weights. Direct GGUF selection calibrates two recognized attention projections. The manifest controls up to eight matrices and also supports Q3 through the CLI. GSQ writes standard Q2_K, Q3_K, Q4_K, Q6_K or Q8_0 tensor variants. The optimization mode disables both target selectors and requires a supported AMD Polaris driver. The capture worker uses an AVX2 host build. [GSQ](https://github.com/IST-DASLab/GSQ) · [RCO](https://github.com/IST-DASLab/RCO)

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
