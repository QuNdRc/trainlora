# Плагинист: API движка и структура датасета

Цель: дать нейронке достаточно данных об API, чтобы она сгенерировала LoRA-датасет
для роли «плагинист» (`pluginist`) — бота, который превращает план в конкретные
команды вызова движка.

Источники: `Solution2/DynamicLibrary1/api.h`, `api.cpp`, `grid_world.h`,
`world.h`, `WpfApp1/NativeWorldAPI.cs`.

---

## 1. Роли адаптеров (Multi-LoRA)

| Роль | Поле `lora` | Что делает |
|------|-------------|------------|
| `planner`   | planner   | Планировщик: выдаёт высокоуровневые фикс-команды (`move_target`/`swap`/`add_door`) |
| `expert`    | expert    | Нормконтролёр: проверяет планировку на СП 54.13330.2022, выдаёт нарушения |
| `pluginist` | pluginist | Плагинист: **превращает план в конкретные вызовы нативного API движка** (никкоманды примитивов) |

Плагинист — это «исполнитель»: вход — задача/план (сетка полей, желаемые перемещения,
двери), выход — последовательность примитивных команд движка, которые реально можно
вызвать через C-API.

---

## 2. Модель данных: RoomType

```cpp
enum class RoomType : uint8_t {
    Undefined = 0,
    LivingRoom = 1,   // Гостиная
    Kitchen = 2,      // Кухня
    Bathroom = 3,     // Санузел
    Corridor = 4      // Коридор
};
```

---

## 3. Два типа мира

### 3.1. Непрерывный мир (WorldState, `CreateWorld`)
Комнаты — прямоугольники с float-координатами, физика устранения наложений.
- `AddRoom(world, x, y, w, h, tx, ty, roomType)` -> id
- `MoveRoomTarget(world, id, tx, ty)` — двигать цель комнаты
- `SwapRooms(world, idA, idB)` — поменять позиции двух комнат
- `RemoveRoom(world, id)`
- `AddDoor(world, roomA, roomB, doorX, doorY)` -> door_id
- `RemoveDoor(world, door_id)`
- `StepWorld(world, dt, minX, minY, maxX, maxY)` — физика
- `IsReachable(world, from, to)`, `GetRoomPath(from,to)`, `GetCorridorLength`

### 3.2. Сеточный мир (GridWorldState, `CreateGridWorld`) — ОСНОВНОЙ
Дискретная «шахматная» модель. Комнаты — произвольные фигуры из клеток.
Создание:
- `CreateGridWorld(capacity, minX, minY, maxX, maxY, cellSize, seed)`
- `AddGridRoom(world, pivotX, pivotY, localCells[], count, roomType, priority)`
  - `localCells` — плоский массив пар `[x0,y0, x1,y1, ...]`
- `RemoveGridRoom(world, id)`

Примитивы перемещения/поворота:
- `ShiftRoom(world, roomId, dx, dy)` — сдвиг на (dx, dy) клеток
- `RotateRoom(world, roomId, angleQuarters)` — 0/1/2/3 (0°..270° по часовой)
- `MoveGridRoomTo(world, roomId, pivotX, pivotY)` — абсолютный pivot (drag / AI)

Сетка и коллизии:
- `SnapshotRoomsToGrid(world)` — переносить комнаты в сетку
- `ResolveAllCollisions(world, maxIterations)` — устранить наложения
- `EvaluatePositionScore(world, roomId, px, py)` — очк координаты
- `FindBestNeighbor(world, roomId)` — лучшее соседнее направление (упаковано в int)

Геометрия конвертации:
- `GridToWorldX/Y`, `WorldToGridX/Y`
- `GetGridBounds(&minX,&minY,&maxX,&maxY)`, `GetGridCellSize()`

Зоны и «волна»:
- `SetZoneRect(world, minX,minY,maxX,maxY, zoneTag)`
- `MarkPerimeterStreet(world)`
- `MarkWindowEdge(world, edge, start, end)` (edge 0=right,1=top,2=left,3=bottom)
- `ComputeWaveField(world)`

Формы-хелперы:
- `MakeRectShape(widthCells, heightCells, outCells[], maxCells)` -> кол-во клеток
- `MakeLShape(mainLength, branchLength, branchDir, outCells[], maxCells)`
  - branchDir: 0=right,1=up,2=left,3=down

Стат-строка:
- `GetGridRoomData(...)` — ids, pivots, roomTypes, priorities
- `GetGridRoomCellCount(roomId)`, `GetGridRoomCells(roomId, outCells[], max)`
- `GetGridWorldStateString(world)`

---

## 4. ZoneTag (битовая маска клетки)

```cpp
enum ZoneTag : uint8_t {
    ZONE_NONE      = 0,
    ZONE_STREET    = 1 << 0,  // улица / внешняя стена
    ZONE_WINDOW    = 1 << 1,  // зона окна
    ZONE_WET       = 1 << 2,  // мокрая зона (санузел/кухня)
    ZONE_PILLAR    = 1 << 3,  // несущая колонна (непроходимо)
    ZONE_ENTRANCE  = 1 << 4,  // вход в квартиру
    ZONE_INTERIOR  = 1 << 5,  // внутренняя стена
    ZONE_CORRIDOR  = 1 << 6,  // предпочтительный коридор
};
```

---

## 5. Формат входа (что подаём модели)

Пример пользовательской задачи для плагиниста (input, структура близка к
`build_user_prompt()` + желаемая операция):

```
Текущая планировка (grid):
- Room r0: Kitchen, pivot (4, 8), cells [[-1,0],[0,0],[1,0],[2,0],[0,1],[1,1]], priority 5
- Room r1: Bathroom, pivot (8, 8), cells [[0,0],[1,0],[0,1],[1,1]], priority 3
- ...
[CONFLICTS]r0,r1;...
[DOORS]d0,r0,r2,5,9;...

Задача: сдвинуть кухню так, чтобы она не наезжала на санузел, и добавить дверь между кухней и коридором.
Ответь двухраздельно: 'Решение: ...' + JSON с командами примитивов движка.
```

---

## 6. Формат выхода — двухраздельный (как в остальных ролях)

Формат ответа модели:

```
Решение: <человеческий текст на русском, что плагинист сделает>

{"commands":[{"action":"move_grid_room_to","room_id":"0","pivot_x":5,"pivot_y":8,"reason":"..."},{"action":"add_door","room_a":"0","room_b":"4","door_x":5,"door_y":9,"reason":"..."}],"explanation":"..."}
```

### 6.1. Допустимые `action` для плагиниста (примитивы движка)

| action | Параметры | Нативная функция |
|--------|-----------|------------------|
| `move_target` | room_id, new_tx, new_ty | `MoveRoomTarget` (непрерывный мир) |
| `move_grid_room_to` | room_id, pivot_x, pivot_y | `MoveGridRoomTo` (сеточный мир) |
| `shift_room` | room_id, dx, dy | `ShiftRoom` |
| `rotate_room` | room_id, angle_quarters | `RotateRoom` |
| `swap` | id_a, id_b | `SwapRooms` |
| `add_door` | room_a, room_b, door_x, door_y | `AddDoor` |
| `remove_room` | room_id | `RemoveGridRoom` / `RemoveRoom` |
| `add_grid_room` | pivot_x, pivot_y, local_cells[], room_type, priority | `AddGridRoom` |
| `resolve_collisions` | max_iterations | `ResolveAllCollisions` |

Правила валидности (согласованно с `command_validator.py`):
- максимум 3 команды за ответ;
- все id должны существовать;
- координаты/пивоты в границах;
- `swap`/`add_door` — разные id.

---

## 7. Формат записи в датасет (для train_lora.py)

Каждый пример — объект:
```json
{
  "instruction": "Преврати план в команды примитивов движка.",
  "input_lines": [
    "Текущая планировка (grid): ...",
    "Задача: ..."
  ],
  "output_text": "Сдвигаю кухню, добавляю дверь к коридору.",
  "output": {
    "commands": [
      {"action":"move_grid_room_to","room_id":"0","pivot_x":5,"pivot_y":8,"reason":"устранить наложение с санузлом"},
      {"action":"add_door","room_a":"0","room_b":"4","door_x":5,"door_y":9,"reason":"связать кухню с коридором"}
    ],
    "explanation": "Кухня наезжает на санузел — сдвигаю pivot и добавляю дверь."
  }
}
```

Поля: `instruction` (роль), `input_lines` (список строк входа — планировка/код/задача),
`output_text` (человеческое пояснение), `output` (JSON-блок команд).

---

## 8. Примечания для генератора

1. **Использовать `input_lines`** — это поле для планировки/ТЗ/контекста API.
2. **Двухраздельный ответ** — обязателен: `Решение: ...` + `\n` + JSON-блок.
3. **Только валидные команды** — id/координаты/пивоты должны быть согласованы с входом.
4. **Разнообразить**: `move_grid_room_to`, `shift_room`, `rotate_room`, `add_door`,
   `remove_room`, `add_grid_room`, `resolve_collisions`, смешанные (2–3 команды).
5. **Room type**: 1=LivingRoom, 2=Kitchen, 3=Bathroom, 4=Corridor.
6. **Angle**: 0/1/2/3 = 0°/90°/180°/270° (по часовой).
7. **Coordinates**: сеточные (в клетках) против мира (в float-юнитах) не путать.