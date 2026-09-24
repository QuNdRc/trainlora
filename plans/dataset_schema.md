# Схема датасета для LoRA-обучения планировщика

Контракт программы: `C:/Users/Даниил/RiderProjects/Solution2/ai_orchestrator`

## 1. Что реально ожидает программа

Запрос к LLM строится в [`prompt_builder.build_user_prompt()`](../../C:/Users/Длянил/RiderProjects/Solution2/ai_orchestrator/prompt_builder.py:40)
и выглядит как человекочитаемый блок:

```
Current apartment layout:
========================================
- Room r0: Living Room, position (0.0, 0.0), size 6.0 x 5.0
- Room r1: Kitchen, position (5.0, 5.0), size 3.0 x 3.0
...

Apartment bounds: x in [0.0, 12.0], y in [0.0, 10.0]

Detected conflicts (overlapping rooms):
----------------------------------------
- Conflict between Living Room (id r0) and Kitchen (id r1), overlap depth 1.5

Doors (room connectivity):
----------------------------------------
- Door d0: connects Living Room (id r0) and Hall (id r4), at (3.0, 2.0)

Relevant SNiP building norms (Техэксперт context to follow):
----------------------------------------
<выдержка из snip_rules.txt>

Respond with valid JSON only. Three allowed command shapes: ... Wrap them in a JSON object: {"commands":[...],"explanation":"<text>"}.
```

### Формат ответа (контракт)

Программа принимает **только** одно из:

1. **Объект** (основной, заявлен в промпте):
```json
{
  "commands": [
    {
      "action": "move_target",
      "room_id": "r1",
      "new_tx": 4.2,
      "new_ty": 3.0,
      "reason": "Сдвигаю кухню, чтобы устранить пересечение с гостиной"
    }
  ],
  "explanation": "Кухня пересекает гостиную; сдвигаю её к границе коридора"
}
```

2. **Список** команд «голышом» (тоже принимается `_parse_llm_response`):
```json
[{"action":"swap","id_a":"r1","id_b":"r2","reason":"..."}]
```

### Жёсткие ограничения валидатора ([command_validator.py](../../C:/Users/Длянил/RiderProjects/Solution2/ai_orchestrator/command_validator.py:17))

| Поле | Правило |
|------|---------|
| `action` | только `move_target`, `swap`, `add_door` |
| `move_target` | `room_id` обязан существовать; `new_tx`/`new_ty` числовые и внутри границ |
| `swap` | `id_a != id_b`, оба существуют |
| `add_door` | `room_a != room_b`, оба существуют; `door_x`/`door_y` в границах |
| количество | SYSTEM_PROMPT: максимум 3 команды за ответ |

Любая команда, нарушившая правило, **молча отбрасывается** (не ломает запрос, но и не выполняется). Значит датасет обязан давать только валидные примеры.

## 2. Формат примеров датасета

Каждый пример — связка «вход программы → ожидаемый выход».

```json
{
  "instruction": "Анализируй планировку и предложи исправления командами.",
  "input_lines": [
    "Current apartment layout:",
    "========================================",
    "- Room r0: Living Room, position (0.0, 0.0), size 6.0 x 5.0",
    "- Room r1: Kitchen, position (4.5, 4.0), size 3.0 x 3.0",
    "",
    "Apartment bounds: x in [0.0, 12.0], y in [0.0, 10.0]",
    "",
    "Detected conflicts (overlapping rooms):",
    "----------------------------------------",
    "- Conflict between Living Room (id r0) and Kitchen (id r1), overlap depth 1.0"
  ],
  "output": {
    "commands": [
      {
        "action": "move_target",
        "room_id": "r1",
        "new_tx": 5.5,
        "new_ty": 4.0,
        "reason": "Кухня пересекает гостиную; сдвигаю её вправо, за пределы перекрытия"
      }
    ],
    "explanation": "Обнаружено пересечение кухни и гостиной. Сдвигаю кухню, сохраняя остальные комнаты."
  }
}
```

### Почему `input_lines` — не единый `text`

Программа подаёт промпт дословно через `build_user_prompt()`. Чтобы LoRA выучил **реальный вход**, примеры должны максимально повторять его структуру. Поэтому поля:

- `input_lines` (или `instruction` + `input`) — воспроизводит блок, который генерирует программа.
- `output` — JSON-объект `commands` + `explanation` (валидный по контракту).

## 3. Правила генерации примеров (чтобы LoRA не выучил мусор)

1. **Только валидные команды.** Каждый `room_id`, `id_a`, `id_b`, `room_a`, `room_b` обязан существовать во входе; координаты — внутри границ; `swap`/`add_door` — разные id.
2. **Всегда оборачивать в `{"commands":[...],"explanation":"..."}`** (объект, а не список) — промпт это прямо требует, консистентность важна.
3. **Максимум 3 команды** — как требует SYSTEM_PROMPT.
4. **`reason` — на русском, но форма контракта — точно как в `Command`.** Программа кладёт `reason` как строку; язык внутри произвольный.
5. **Разнообразие сценариев:**
   - `move_target` — устранение пересечения комнат;
   - `swap` — перестановка комнат местами (например, кухня/санузел);
   - `add_door` — добавление связи между комнатами (если нужен доступ);
   - смешанные случаи (2–3 команды разного типа в одном ответе);
   - случай «нарушений нет» → `{"commands":[],"explanation":"Нарушений не обнаружено"}` (программа такие запросы вообще не шлёт в LLM, но для «expert»-адаптера может понадобиться).
6. **Типы комнат** — из `ROOM_TYPE_NAMES`: `Living Room`, `Kitchen`, `Bathroom`, `Corridor`, `Undefined`; id — строки (`r0`, `r1`, ...).
7. **Формат чисел** — координаты/размеры с одним знаком после запятой (`6.0`, `4.5`), как в `build_user_prompt()`.
8. **Контекст СП (необязательно)** — для ответов «expert»-стиля включать выдержку норм в `input_lines`, чтобы связать нарушение с конкретным пунктом `snip_rules.txt`.

## 4. Токенизация (изменение в train_lora.py)

Текущий скрипт ждёт поле `text` и режет всё в одну последовательность. Под новый формат нужно:

- читать `instruction` + `input_lines` для входа;
- `output` сериализовать в строку (`json.dumps(output, ensure_ascii=False)`) как целевую последовательность;
- собирать по чат-шаблону токенизатора: системный промпт + user(вход) → ответ = output;
- target/labels маскировать так, чтобы модель училась только на части ответа (зависит от `apply_chat_template` / `train_on_completions_only`), иначе модель запомнит и промпт, но это тоже допустимо для LoRA.

## 5. Что НЕ входит в целевую схему (учитывать на следующем шаге)

- **Формат адаптера (PEFT → GGUF):** текущий `train_lora.py` даёт safetensors-адаптер, а программа на llama.cpp ждёт GGUF/`lora`. Это отдельная задача после согласования схемы датасета.
- **`expert`-адаптер (нормконтролёр):** твой пример с `{"status":"violations","items":[...]}` — это скорее формат для `expert`, а не `planner`. Для планировщика формат другой (см. выше). Решить, какой адаптер тренируем первым.