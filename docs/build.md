# Build and tests

| Path | Contents |
| --- | --- |
| [`app/`](../app/) | CLI, GUI, conversion and calibration. [`app/vulkan/`](../app/vulkan/) contains shaders and the native capture build. |
| [`vendor/hf/`](../vendor/hf/) | Pinned Hugging Face converter and GGUF reader, shared by source and portable builds. |
| [`portable_build.py`](../portable_build.py), [`Launcher.cs`](../Launcher.cs) | Package the runtime and start it from a temporary directory. |

From the repository root, use Python 3.12 or newer and the Windows .NET Framework C# compiler. Dependencies are verified against [`dependencies.lock.json`](../dependencies.lock.json) and cached in `build-assets/`. The EXE is written to `dist/`.

```powershell
py -3 -I -B portable_build.py
.\dist\GGUF-Converter.exe --cli --help | Out-String
```

For Vulkan calibration, put the MinGW-w64 `bin` directory on `PATH`. Pass `--cxx`, `--glslc`, `--vulkan-include` and `--vulkan-import` with paths to `g++.exe`, Vulkan SDK `glslc.exe`, the SDK include directory and `vulkan-1.lib`. To include the media worker, add `--native-capture` pointing to the output of [`app/vulkan/build_capture.py`](../app/vulkan/build_capture.py). See the [pinned sources and notices](../NOTICE.md).

To smoke test the source without downloading dependencies:

```powershell
py -3 -I -B tests/test_hardware.py
py -3 -I -B tests/test_routes.py
py -3 -I -B -X utf8 app/convert.py --help
```

For actual source-mode conversions, use a Python environment with the pinned packages and unpack the matching CPU tools into `backends/llama/` and `backends/sd/`. For source-mode calibration, place the compiled shaders and helper in `app/vulkan/` and the optional capture worker in `app/capture-runtime/`.
