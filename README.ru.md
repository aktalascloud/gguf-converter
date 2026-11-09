# Конвертер GGUF для Windows

[English](README.md) · [Скачать портативный EXE](https://github.com/aktalascloud/gguf-converter/releases/latest)

Откройте `GGUF-Converter.exe`, выберите файл весов или папку Hugging Face, укажите **Q8_0** или **Q4_0** и нажмите «Конвертировать». Файл или папку можно перетащить на EXE. Для запуска нужны Windows 10/11 x64 и процессор с AVX2. Python, Tk и CPU-конвертеры входят в один файл. При открытии окна они распаковываются во временную папку.

| Вход | Поддержка |
|---|---|
| Текстовый GGUF | Переквантование через llama.cpp, включая разделённые GGUF |
| Папка Hugging Face | Поддерживаемые архитектуры с конфигурацией, токенизатором и весами. FP32, FP16, BF16, поддерживаемый FP8, GPTQ, AWQ GEMM 4-bit с zero points, поддерживаемые схемы compressed-tensors и bitsandbytes NF4/FP4 или построчный INT8 |
| Диффузионные веса | GGUF, Safetensors и CKPT через stable-diffusion.cpp. FP8 E4M3/E5M2 с масштабами и Comfy INT8 tensorwise в Safetensors. Метка Wan сохраняется или определяется по именам тензоров. |
| Artius MOSS | Comfy INT8 и W4A8. Для диффузора и текстового энкодера есть отдельные режимы. |

Результат записывается в отдельный GGUF с защитой от перезаписи существующих файлов. Часть тензоров сохраняет точность с плавающей запятой. Для Q4_0 из папки Hugging Face требуется место под промежуточный FP32 GGUF. Для некоторых диффузионных весов с отдельными масштабами также требуется временное место на диске результата.

Пример запуска из PowerShell.

```powershell
Start-Process -FilePath '.\GGUF-Converter.exe' -ArgumentList '--cli "D:\Модели\Модель" -q Q4_0 -t 8' -Wait -PassThru
```

Сборка из исходников требует Python 3.12+ на компьютере сборки. Команда проверяет SHA зависимостей и сохраняет EXE в `dist/`.

```powershell
py -3.12 -I -B portable_build.py
```

[Лицензия](LICENSE.md) · [Исходники и лицензии компонентов](NOTICE.md).
