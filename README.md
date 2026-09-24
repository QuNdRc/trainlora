# LoRA-файнтюнинг пайплайн для квартирного планировщика

Доменно-специфичные LoRA-адаптеры, обученные на Gemma-4 (4.6B) для управления
движком расстановки комнат через структурированные JSON-команды — от датасета
до боевых GGUF-LoRA-файлов для llama.cpp.

## Что в репозитории

Полный, самодостаточный пайплайн, который берёт обучающие примеры, файнтюнит
**4.6-миллиардную языковую модель** (Gemma-4-E2B-it) на **чистом CPU**,
и выдаёт крошечные (~112 МБ) LoRA-адаптеры, готовые к деплою в llama.cpp.
Адаптеры генерируют команды исправления планировки
(`move_target`, `swap`, `add_door`) в строгом двухраздельном формате:
человекочитаемое рассуждение, затем машиночитаемый JSON.

Три адаптера покрывают полный цикл планирования:

| Адаптер | Роль | Датасет |
|---------|------|---------|
| `expert` | Нормконтролёр — проверяет и правит расстановку комнат по нормам | `dataset_norm.json` |
| `planner` | Планировщик — генерирует планировки с нуля | `dataset_planer.json` |
| `pluginist` | Исполнитель — переводит текстовые указания в команды движка | `dataset_pluginist.json` |

## Архитектура

```
dataset.json  ──→  train_lora.py  ──→  *_lora_ready/
  (JSON)             (PEFT/HF)            (safetensors)

  *_lora_ready/  ──→  convert_adapters_to_gguf.py  ──→  *.gguf
  (safetensors)        (llama.cpp subprocess)             (GGUF-LoRA)

  *.gguf  ──→  llama-server --lora *.gguf  ──→  HTTP API
```

- **Модель**: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) — только текстовая часть (Gemma4ForCausalLM)
- **Обучение**: PEFT LoRA (r=16, alpha=32) на 7 целевых модулях в каждом слое
- **Железо**: CPU, 16+ ГБ ОЗУ, веса в bfloat16 (~14 ГБ)
- **Формат ответа**: двухраздельный — `"Решение: …\n{JSON}"` — лосс замаскирован на токенах промпта (-100)

## Быстрый старт

### 1. Окружение

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers==5.15.1 peft==0.20.0 datasets safetensors
```

Скачай веса Gemma-4 (safetensors) в `C:/ai/weight/model.safetensors`.

### 2. Обучение адаптера

```bash
python train_lora.py --dataset dataset_norm.json --output ./expert_lora_ready --adapter-name expert
```

По одной команде на адаптер. Обучение идёт около 13 секунд на шаг
на современном CPU — примерно 1 час на датасет из 50 примеров × 5 эпох.

### 3. Конвертация в GGUF-LoRA

```bash
python convert_adapters_to_gguf.py --adapter expert_lora_ready
```

Требуются исходники [llama.cpp](https://github.com/ggml-org/llama.cpp)
с пакетами `conversion/` и `gguf-py/` в `C:/ai/llama-src/llama.cpp-master`.

### 4. Запуск в llama.cpp

```bash
llama-server.exe -m gemma-4-base.gguf --lora expert_lora.gguf --port 8080
```

Либо горячая замена адаптеров на лету через `POST /lora-adapters`.

## Файлы

| Файл | Назначение |
|------|------------|
| [`train_lora.py`](train_lora.py) | Основной скрипт обучения — сборка модели, загрузка весов, PEFT |
| [`convert_datasets.py`](convert_datasets.py) | Предобработка: разбиение сырых ответов на текст + JSON |
| [`convert_adapters_to_gguf.py`](convert_adapters_to_gguf.py) | Пакетный конвертер: PEFT safetensors → GGUF-LoRA |
| [`bench_step.py`](bench_step.py) | Микробенчмарк: замер секунд на шаг обучения |
| `dataset_*.json` | Готовые обучающие датасеты |
| `plans/` | Архитектурная документация, дизайн пайплайна, заметки |

## Формат ответа

Модель обучена выдавать двухраздельный ответ:

```
Решение: Кухня пересекает гостиную в точке (3.2, 1.5), сдвигаю её вправо.
{"commands":[{"action":"move_target","room_id":"r1","new_tx":5.5,"new_ty":4.0}],"explanation":"…"}
```

Такое разделение позволяет модели «думать вслух» (цепочка рассуждений —
для отладки и человеческого контроля) и одновременно выдавать
машиночитаемый JSON (парсится движком планировщика напрямую).

## Лицензия

MIT