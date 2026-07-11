# Схема импорта 0.4.1a1

## Каноническая таблица

Обязательные поля:

| Поле | Смысл |
|---|---|
| `test_id` | устойчивый идентификатор испытания |
| `stage` | номер/метка ступени; пустое значение сохраняется и создаёт warning |
| `load` | исходная сила или давление согласно metadata |

Требуется `settlement` либо хотя бы один `indicator_1..4`. Допустимы `branch`, `timestamp`, `reference_indicator`, `horizontal_indicator`, `status`, `comment`, `group`, `pair_id`.

Порядок строк является физическим порядком протокола. Импорт не сортирует данные и не добавляет `(0;0)`.

## Происхождение точки

Excel-адаптер добавляет:

```text
sheet_name
source_row
source_columns
sequence_index
raw_stage, raw_load, raw_indicator
parsed_stage, parsed_load, parsed_indicator
load_unit
```

`raw_cells.csv` содержит отдельную запись для каждой использованной ячейки: лист, строку, букву столбца, исходное значение, распознанное значение и каноническое поле.

## Режимы

- `strict`: известные алиасы разрешены, неизвестные и неоднозначные заголовки блокируют расчёт.
- `interactive`: источники задаются JSON-объектом `{canonical_field: column_letter_or_header}`.
- `heuristic`: совместимость с legacy-блоками `N испытание`; всегда создаёт warning. Показания legacy-индикатора остаются raw до отдельного подтверждения его шкалы.

В интерфейсе interactive-mapping сначала показываются канонические строки и координаты raw-ячеек. Подтверждение имеет область действия `input SHA-256 + sheet + header_row + mapping` и сбрасывается при изменении любого компонента.

XLSX проходит ZIP/XML preflight и лимиты ресурсов до `openpyxl`. Повреждённый/подозрительный контейнер возвращает блокирующий `ValidationIssue`, исходные bytes не изменяются.

## Контракт ошибки

Каждая проблема экспортируется с полями:

```text
severity/level, code, message, test_id, sheet, row, column,
raw_value, suggested_action, blocks_processing
```

## Metadata нового проекта

Во всех режимах физические параметры должны быть заданы явно; `heuristic` ослабляет только распознавание legacy-схемы:

```text
load_kind, load_unit, load_factor, lever_ratio,
settlement_unit, indicator_resolution_mm,
stamp_diameter_mm или stamp_area_m2
```

`load_zero` рекомендуется указывать явно; при отсутствии создаётся warning.

Непустой объект `metadata.tests` трактуется как полный реестр опытов выбранного проекта, а не как произвольное частичное множество overrides. Каждый protocol `test_id` должен присутствовать в нём; поле можно опустить полностью, если индивидуальные параметры не нужны.

Если строка не содержит прямой `settlement` и использует `indicator_*`, предпочтителен поканальный паспорт:

```text
indicator_passports.<channel>:
  type, serial_number, range_mm, division_mm,
  correction_factor, verification_date, verification_valid_until,
  mode, initial_reading, zero_correction_mm, max_increment_mm
```

Режимы `increasing`, `increasing_wrapped`, `decreasing`, `decreasing_wrapped` и `cumulative_settlement` поддержаны. Для нескольких оборотов между соседними строками используется явный `indicator_N_turn_number`; без него неоднозначность блокирует расчёт. Старые общие `indicator_*` metadata совместимы только с уже преобразованной накопленной осадкой. При прямой `settlement` неподтверждённые индикаторы не влияют на осадку и крен.

Формула силы:

```text
F_kN = convert((load_raw - load_zero) * load_factor * lever_ratio, load_unit -> kN)
p_kPa = F_kN / A_m2
```

## Миграция с 0.4.0

- CSV остаётся совместимым; к таблице добавляются source/raw/parsed столбцы.
- CLI принимает CSV и XLSX, добавлены `--import-mode`, `--column-map`, `--sheet`, `--header-row`.
- ZIP теперь содержит исходный файл, `provenance.json` и `manifest.json`.
- `provenance.json` фиксирует commit, dirty-state, SHA-256 дерева исходников и версии `openpyxl`/`defusedxml`.
- Для нового строгого проекта metadata, ранее получавшие скрытые defaults, должны быть заполнены явно.
