# GGUF Converter for Windows

[English](README.md) · [Скачать](https://github.com/aktalascloud/gguf-converter/releases/tag/v1.9.0) · [Документация](docs/calibration.md) · [Данные](https://github.com/aktalascloud/gguf-converter/blob/28aa603895082e0f42170ecda1dbc69e5acd598a/benchmarks/2026-11-17-whole-gsq-rco-rx580.json)

Конвертация текстовых и медиамоделей в GGUF из одного портативного EXE для Windows с Python внутри. Перед конвертацией можно оценить размер, а поддерживаемые веса калибровать на AMD Polaris через Vulkan.

**Запуск.** Запустите `GGUF-Converter.exe`, выберите GGUF, Safetensors или папку модели Hugging Face и укажите разрядность. Если есть исходный чекпойнт, используйте его для калибровки.

**Проверка.** В замере GSQ/RCO от 17 ноября 2026 года на RX 580 ошибка проекций UMT5-XXL снизилась с 60 640 до 37 930 на записанных активациях в заданном бюджете тензоров. [Опубликованный энкодер](https://huggingface.co/TalasCloud/UMT5-XXL-Encoder-GSQ-RCO-GGUF) весит 3,62 ГБ. Ошибка проекций характеризует подгонку.

## Калибровка

| Переключатель | Для чего |
| --- | --- |
| **Выкл** | Обычная конвертация. |
| **Авто** | ACTCAL для текущего кванта или GSQ/RCO для другой разрядности. |
| **ACTCAL** | Подбор кодов или масштабов при сохранении типов тензоров. |
| **GSQ/RCO** | Подбор сочетания типов тензоров под бюджет размера. |
| **MACaQ-R** | Уточнение кодов в текущем GGUF с фиксированными масштабами. |

Калибровка работает через Vulkan на поддерживаемых AMD Polaris. Выбирайте исходный чекпойнт той же версии. При его отсутствии ACTCAL проверяет обратную конвертацию относительно F32 из текущего GGUF. Манифест задаёт метод калибровки. Для фото, видео и YuE2 активации может собрать подходящий профиль KoboldCpp. [Инструкция](docs/calibration.md)

Встроенная адаптация MACaQ-R использует гессиан блоками по 32 столбца и штатные ядра генерации GGUF. [GSQ](https://github.com/IST-DASLab/GSQ) · [RCO](https://github.com/IST-DASLab/RCO) · [MACaQ](https://github.com/huawei-csl/MACaQ)

[Замеры на RX 580](benchmarks/README.md) · [Патчи](patches/README.md) · [Лицензии и компоненты](NOTICE.md)

[Все проекты](https://github.com/aktalascloud)
