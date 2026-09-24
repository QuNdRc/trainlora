# Команды запуска обучения LoRA (Gemma-4-E2B-it)

Все команды выполняются из папки проекта:

```
cd /d C:\Users\Даниил\RiderProjects\Solution3
```

> Перед запуском: закрой Rider и лишние программы, отключи автоперекорку Mem Reduct.

---

## 1. Обычный запуск (в окне консоли, видно прогресс)

### Эксперт / нормконтролёр
```
python train_lora.py --dataset dataset_norm.json --output ./expert_lora_ready --adapter-name expert
```

### Планировщик / генератор
```
python train_lora.py --dataset dataset_planer.json --output ./planner_lora_ready --adapter-name planner
```

### Плагинист / исполнитель
```
python train_lora.py --dataset dataset_pluginist.json --output ./pluginist_lora_ready --adapter-name pluginist
```

---

## 2. Запуск в фоне с лог-файлом (можно закрыть окно)

### Эксперт
```
start /b python train_lora.py --dataset dataset_norm.json --output ./expert_lora_ready --adapter-name expert > train_log.txt 2>&1
```

### Планировщик
```
start /b python train_lora.py --dataset dataset_planer.json --output ./planner_lora_ready --adapter-name planner > train_log.txt 2>&1
```

### Плагинист
```
start /b python train_lora.py --dataset dataset_pluginist.json --output ./pluginist_lora_ready --adapter-name pluginist > train_log.txt 2>&1
```

### Посмотреть прогресс/ошибки
```
type C:\Users\Даниил\RiderProjects\Solution3\train_log.txt
```

---

## 3. Запуск в отдельном окне
```
start "train" python train_lora.py --dataset dataset_norm.json --output ./expert_lora_ready --adapter-name expert
```

---

## 4. Замер скорости 1 шага (тест)
```
python bench_step.py --dataset dataset_norm.json --steps 1
```

Старой замер (Qwen): 1 шаг ≈ 12.97 с → ~1 ч на 50 примеров × 5 эпох.
Gemma-4 (4.6B) — возможно на 10-20% дольше из-за большего числа слоёв (35 vs 32)
и большого `embed_tokens_per_layer` (4.7 ГБ дополнительно).

---

## 5. Где лежат готовые адаптеры (после обучения)

| Роль | Папка |
|------|-------|
| expert | `C:\Users\Даниил\RiderProjects\Solution3\expert_lora_ready` |
| planner | `C:\Users\Даниил\RiderProjects\Solution3\planner_lora_ready` |
| pluginist | `C:\Users\Даниил\RiderProjects\Solution3\pluginist_lora_ready` |

После обучения: `adapter_config.json` + `adapter_model.safetensors`.

---

## 6. Дальнейшие шаги (после обучения)

1. Сконвертировать адаптер в GGUF: `python convert_adapters_to_gguf.py --adapter expert_lora_ready`
2. Запустить llama.cpp-сервер с `--lora`.
3. Правка parse-фикса программа под двухраздельный ответ: `plans/tz_ai_orchestrator_refactor.md`.

---

## 7. Особенности Gemma-4

- **Модель**: `google/gemma-4-E2B-it`, 4.6B параметров, 35 слоёв
- **Архитектура**: MQA (1 KV head, 8 query heads), double-wide MLP, per-layer gating
- **Уникальная фича**: `embed_tokens_per_layer` (262144×8960) — отдельные эмбеддинги на каждый слой, занимают ~4.7 ГБ
- **KV shared layers**: `num_kv_shared_layers=20` — последние 20 слоёв разделяют KV-проекции
- **Веса**: лежат в `C:/ai/weight/model.safetensors` (10.2 ГБ, мультимодальная), префикс текстовой части — `model.language_model.`
- **Конвертер в GGUF**: llama.cpp должен поддерживать Gemma-4 (b10063+). Если нет — адаптеры остаются в формате safetensors для PEFT-загрузки через transformers