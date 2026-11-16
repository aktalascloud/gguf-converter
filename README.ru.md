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

## GSQ-RCO на Vulkan

Выберите **GSQ-RCO-Q2**, **GSQ-RCO-Q3** или **GSQ-RCO-Q4** и откройте манифест `gguf-gsq-rco-vulkan-v1`. Портативный EXE калибрует веса методом Gumbel-Softmax и запускает поиск RCO на AMD Polaris с Vulkan 1.1. Итоговый GGUF содержит стандартные тензоры Q2_K, Q3_K и Q4_K. Для GPU нужен драйвер AMD с Vulkan. [Пример манифеста](README.md#vulkan-gsq-rco-preview) · [GSQ](https://github.com/IST-DASLab/GSQ) · [RCO](https://github.com/IST-DASLab/RCO)

Укажите от 2 до 32 записанных активаций для каждой матрицы в массиве NumPy `.npy` формы `[samples, input_width]`. Можно выбрать до восьми двумерных тензоров с шириной, кратной 256. RCO использует нормализованную ошибку реконструкции слоёв. Бюджет в байтах учитывает и исходные тензоры вне списка. Название режима выбирает цель из манифеста, фактические типы и размеры определяются бюджетом. Сбор активаций и оптимизация по ошибке модели целиком выполняются отдельно.

## Сборка GSQ-RCO

Выберите **GSQ-RCO-Q2**, **GSQ-RCO-Q3** или **GSQ-RCO-Q4** и откройте JSON-манифест. Конвертер побайтно копирует упакованные тензоры из заранее оптимизированных GSQ-файлов по карте назначений RCO. Калибровка GSQ и поиск RCO выполняются отдельно. Портативный EXE собирает итоговый GGUF локально. [GSQ](https://github.com/IST-DASLab/GSQ) · [RCO](https://github.com/IST-DASLab/RCO)

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

Пути задаются относительно манифеста. Добавьте нужные цели в targets. Значение каждого назначения указывает на файл из variants. Остальные тензоры берутся из базового файла. Необязательный max_tensor_bytes ограничивает общий размер всех тензоров. Конвертер выводит фактическое число бит на параметр.

```powershell
Start-Process -FilePath '.\GGUF-Converter.exe' -ArgumentList '--cli "D:\Модели\Модель" -q Q4_0 -t 8' -Wait -PassThru
```

Сборка из исходников требует Python 3.12+ на компьютере сборки. Команда проверяет SHA зависимостей и сохраняет EXE в `dist/`.

```powershell
py -3.12 -I -B portable_build.py
```

Для Vulkan-сборки передайте `--cxx`, `--glslc`, `--vulkan-include` и `--vulkan-import`. Укажите `g++.exe` из MinGW-w64, `glslc.exe`, каталог заголовков Vulkan SDK и `vulkan-1.lib`. В портативный релиз входят готовый помощник и шейдеры SPIR-V.

[Лицензия](LICENSE.md) · [Исходники и лицензии компонентов](NOTICE.md).
