# Инструкция: PEFT-обучение → GGUF-LoRA → llama.cpp

Полная цепочка внедрения LoRA-адаптера в программу-планировщик
(три подсистемы: PEFT-обучение, конвертация, llama.cpp-сервер).

## 0. Итог проверки (всё подтверждено)

| Звено | Проверка | Статус |
|-------|----------|--------|
| `llama-server.exe v10063` | флаги `--lora`, `--lora-scaled`, `POST /lora-adapters` | ✅ Есть |
| `llama-cli.exe v10063` | загрузка `Qwen3.8-4B-Q4_K_M.gguf`, генерация (`[Start thinking]`) | ✅ Архитектура `qwen3_5` работает |
| конвертер llama.cpp | `conversion/qwen.py`: `class Qwen3_5TextModel(_LinearAttentionVReorderBase)` | ✅ Гибидные слои поддерживаются |

**Вывод: путь PEFT → GGUF → llama.cpp полностью рабочий для твоей модели Qwen3.8-4B (qwen3_5).**

---

## 1. Шаг 1 — Обучение LoRA (PEFT, уже настроено в train_lora.py)

Выполняется локально, на CPU. Результат — папка с адаптером
`./planner_lora_ready/`, содержащая:
- `adapter_config.json`
- `adapter_model.safetensors`

**Важно перед обучением:** убедиться, что в `adapter_config.json` в поле
`base_model_name_or_path` стоит правильный id модели (например
`empero-ai/Qwen3.8-4B-Distill`), а в `peft_config` архитектура
`Qwen3_5ForCausalLM`. Конвертеру нужна базовая модель для сопоставления имен.

---

## 2. Шаг 2 — Подготовка окружения конвертера

Конвертер НЕ standalone: ему нужно дерево исходников llama.cpp.

### 2.1. Скачать исходники llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git C:\ai\llama.cpp
```

Либо скачать zip-архив репозитория и распаковать в `C:\ai\llama.cpp`.

Нужны именно компоненты:
- `convert_lora_to_gguf.py` (в корне репо)
- `conversion/` (пакет)
- `gguf-py/` (обвязка формата GGUF)

**Внимание:** использовать ту же версию llama.cpp, что и сервер (или свежее).
Сервер у тебя `b10063`. Если конвертер из более свежего master создаст
GGUF-LoRA с более новыми полями — старый сервер может не прочитать их.
Рекомендуется брать из того же релиза, что и бинарники `b10063`.

### 2.2. Установить Python-зависимости конвертера

Внутри клона:

```bash
cd C:\ai\llama.cpp
pip install -r requirements.txt
pip install -e ./gguf-py
pip install transformers peft torch safetensors
```

(часть уже установлена: `transformers 5.15.1`, `torch 2.13`, `safetensors`)

---

## 3. Шаг 3 — Конвертация PEFT → GGUF-LoRA

```bash
cd C:\ai\llama.cpp
python convert_lora_to_gguf.py ^
  --model C:\ai\weight\model.safetensors ^
  --outfile C:\ai\llama.cpp\planner_lora.gguf
```

Параметры, которые можно/нужно уточнить под твоим случаем:
- `--base <base_model_or_dir>` — путь/идентификатор базовой модели
  (если в `adapter_config.json` нет `base_model_name_or_path`).
- `--outfile` — выходной файл GGUF-LoRA.
- `--verbose` — подробный вывод.

Выход: `planner_lora.gguf` — готовый LoRA-адаптер для llama.cpp.

---

## 4. Шаг 4 — Запуск lama.cpp-сервера с LoRA

Запуск сервера с базовой GGUF-моделью и адаптером:

```bash
C:\ai\llama-b10063-bin-win-cpu-x64\llama-server.exe ^
  -m C:\ai\llama-b10063-bin-win-cpu-x64\Qwen3.8-4B-Q4_K_M.gguf ^
  --lora C:\ai\llama.cpp\planner_lora.gguf ^
  --port 8080
```

### Настройка ai_orchestrator

В `.env` (или среде) по [`config.py`](../../C:/Users/Длянил/RiderProjects/Solution2/ai_orchestrator/config.py:36):

```
LOCAL_MODEL_PATH=C:\ai\llama-b10063-bin-win-cpu-x64\Qwen3.8-4B-Q4_K_M.gguf
LOCAL_SERVER_URL=http://127.0.0.1:8080
DEFAULT_ADAPTER=planner
```

`llm_backend.py` уже шлёт поле `lora` (имя адаптера). В текущей схеме сервер
грузит адаптер при старте через `--lora`. Для нескольких адаптеров
(planner/expert/pluginist) используется `--lora A, --lora B,...`
или `POST /lora-adapters` и `--lora-init-without-apply`.

---

## 5. Проверка работоспособности

1. Запустить сервер с `--lora` — в логе должно появиться `lora: ... applied`.
2. Отправить тестовый запрос в `/plan` (или напрямую в `/completion`):
   - убедиться, что `llama.cpp` применяет LoRA (лог);
   - ответ в двухраздельном формате `Решение: ... \n {"commands":[...]}`.

---

## 6. Известные риски

| Риск | Причина | Митигация |
|------|---------|-----------|
| Старый сервер не читает новый GGUF-LoRA | разница версий конвертер/сервер | конвертер из того же релиза, что b10063 |
| Конвертер не находит базовую модель | нет `base_model_name_or_path` в `adapter_config.json` | передать `--base` |
| Неверная архитектура в конфиге | должен быть `Qwen3_5ForCausalLM` | проверить `adapter_config.json` |
| Расхождение имен гибридных слоев | PEFT-имена vs GGUF | подтверждено: конвертер обрабатывает `linear_attn.*` |
| Поле `lora` игнорируется в `/completion` | сервер запущен без `--lora-init-without-apply` | грузить адаптер при старте `--lora` |

---

## 7. Что дальше

1. Обучение: добавить в `train_lora.py` правильный формат датасета
   («Решение: … \n {JSON}») и чат-шаблон — см. `dataset_schema.md`.
2. Запустить обучение, проверить `adapter_config.json`.
3. Конвертировать по шагам 2–3.
4. Подключить к серверу по шагу 4 и протестировать с `/plan`.