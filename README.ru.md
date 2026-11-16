# Конвертер GGUF для Windows

[English](README.md) · [Скачать портативный EXE](https://github.com/aktalascloud/gguf-converter/releases/latest)

Откройте `GGUF-Converter.exe`, выберите поддерживаемый файл весов или папку Hugging Face и укажите **Q2, Q4, Q6 или Q8**. Тип модели определяется по метаданным и структуре тензоров. Окно показывает распознанный тип, формат GGUF и точный размер готового файла. Python и инструменты конвертации включены в EXE для Windows 10/11 x64.

| Вход | Поддержка |
|---|---|
| Текстовый GGUF | Переквантование через llama.cpp, включая разделённые GGUF |
| Папка Hugging Face | Поддерживаемые архитектуры с конфигурацией, токенизатором и весами. FP32, FP16, BF16, поддерживаемый FP8, GPTQ, AWQ GEMM 4-bit с zero points, поддерживаемые схемы compressed-tensors и bitsandbytes NF4/FP4 или построчный INT8 |
| Диффузионные веса | GGUF, Safetensors и CKPT через stable-diffusion.cpp. FP8 E4M3/E5M2 с масштабами и Comfy INT8 tensorwise в Safetensors. Метка Wan сохраняется или определяется по именам тензоров. |
| Упакованные веса ComfyUI | Поддерживаемые схемы INT8 tensorwise, FP8 с масштабами и W4A8. Схема выбирается автоматически. |

Результат записывается в отдельный GGUF с защитой от перезаписи существующих файлов. Часть тензоров сохраняет точность с плавающей запятой. Для Q4_0 из папки Hugging Face требуется место под промежуточный FP32 GGUF. Для некоторых диффузионных весов с отдельными масштабами также требуется временное место на диске результата.

### Целевой бэкенд

GPU и CPU выбираются независимо. Можно сочетать Vulkan с AVX или CUDA/Pascal P40 с AVX2. Универсальная цель сохраняет `Q2_K`, `Q4_K`, `Q6_K`, `Q8_0`. При выборе Polaris или Pascal, либо AVX/AVX2, Q4 даёт `Q4_0`. NVIDIA CUDA / INT8 выбирает Q8 в окне. Ядра Vulkan, AVX и CUDA выбирает движок при загрузке GGUF. Для создания файла CUDA не требуется. [Перепаковка в GGML](https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt) · [Pascal DP4A](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/)

## GSQ-RCO на Vulkan

Включите **GSQ/RCO (Vulkan)**, выберите GGUF диффузора и задайте промпт. Если энкодер или VAE находятся в отдельных файлах, подключите их через профиль `.kcpps`. Встроенная Vulkan-сборка записывает активации для двух распознанных проекций. Манифест позволяет выбрать до восьми матриц, задать свою схему и запустить Q3 через CLI. Доступны Q2_K, Q3_K, Q4_K, Q6_K и Q8_0. [Пример манифеста](README.md#vulkan-gsq-rco-preview) · [GSQ](https://github.com/IST-DASLab/GSQ) · [RCO](https://github.com/IST-DASLab/RCO)

В манифесте укажите `.npy` с активациями или `null` для автозаписи. Для автозаписи нужен объект `capture` с полем `prompt`. Матрицы должны иметь ширину, кратную 256. Бюджет учитывает все тензоры GGUF. При `model_loss: true` программа выполняет целый тестовый рендер каждого кандидата и выбирает по MSE изображения или кадров. GSQ обучает выбранные матрицы, а RCO ограничивает список конфигураций бюджетом.

Замеры предыдущей версии привязаны к SHA. [Данные v1.2.0 на RX 580](benchmarks/2026-11-16-rx580.json)

В v1.3.0 проверены автозапись ABDM и Wan на RX 580, фото без апскейла, пять кадров 480p и размеры GGUF для четырёх стандартных форматов. [Данные v1.3.0](benchmarks/2026-11-16-v130-rx580.json)

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

Для Vulkan-сборки передайте `--cxx`, `--glslc`, `--vulkan-include` и `--vulkan-import`. Для встроенной автозаписи укажите `--native-capture` с каталогом, собранным через `vulkan/build_capture.py`.

[Лицензия](LICENSE.md) · [Исходники и лицензии компонентов](NOTICE.md).
