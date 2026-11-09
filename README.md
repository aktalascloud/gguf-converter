# GGUF Converter for Windows

[Русский](README.ru.md) · [Download the portable EXE](https://github.com/aktalascloud/gguf-converter/releases/latest)

Convert local model weights to **Q8_0** or **Q4_0** on a Windows 10/11 x64 PC with AVX2. Open `GGUF-Converter.exe`, select the file or Hugging Face folder, choose the format, and start conversion. You can also drop a file or folder onto the EXE. Python, Tk and the CPU conversion tools are bundled and extracted to a temporary folder when the window opens.

| Input | Conversion path |
|---|---|
| Text GGUF | llama.cpp, including sharded models and supported requantization |
| Local Hugging Face folder | Supported architectures with config, tokenizer and weights. FP32, FP16, BF16, supported FP8, GPTQ, AWQ GEMM 4-bit with zero points, supported compressed-tensors layouts and bitsandbytes NF4/FP4 or row-wise INT8 |
| Diffusion weights | GGUF, Safetensors and CKPT through stable-diffusion.cpp. Includes scaled FP8 E4M3/E5M2 and Comfy INT8 tensorwise Safetensors. Wan architecture metadata is preserved or detected from the tensor names. |
| Artius MOSS | Comfy INT8 and W4A8, with separate diffusion and text-encoder modes |

The converter writes a separate GGUF and protects existing files from replacement. Small tensors and tensors with incompatible dimensions may remain in floating point. Q4_0 conversion from Hugging Face needs room for an intermediate FP32 GGUF. Some scaled diffusion checkpoints also need temporary FP32 space on the output drive. Requantizing INT4 to Q8_0 increases the storage precision, while accuracy is limited by the source weights.

To run a conversion from PowerShell, pass `--cli` and wait for the process to exit.

```powershell
Start-Process -FilePath '.\GGUF-Converter.exe' -ArgumentList '--cli "D:\Models\My model" -q Q4_0 -t 8' -Wait -PassThru
```

To build from source, run the following on Windows with Python 3.12+ and the Windows .NET Framework C# compiler available. The build uses the pinned checksums in `dependencies.lock.json` and writes the EXE to `dist/`.

```powershell
py -3.12 -I -B portable_build.py
```

[License](LICENSE.md) · [Corresponding sources and component notices](NOTICE.md).
