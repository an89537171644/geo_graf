# Release readiness — 0.5.0rc1

## Статус кандидата

`0.5.0rc1` — **candidate for engineering acceptance**. Документ не является
сертификатом, подписью инженера, разрешением на эксплуатацию или объявлением
окончательного релиза. Исходная точка TASK 06:
`e0d303478c7e166b5a608bf90c0a351b4a26ec29`; рабочая ветка:
`release/0.5.0rc1`.

## Что вошло в кандидата

| Фаза | Фактический результат до TASK 06 | Статус в base |
|---|---|---|
| 00 | Воспроизводимый CI, demo и семантический verifier | complete |
| 01 | Явный методический контракт и gate основного `E` | complete |
| 02 | Явный `pair_id` и безопасный fallback к независимому анализу | complete |
| 03 | Поканальная метрология, фиксированная агрегация и аудит индикаторов | complete |
| 04 | Явный выбор кривой, общая область, failure/censoring `1.0` | complete |
| 05 A | HTML/XLSX approval report, manifest и воспроизводимый архив | complete |

Эти статусы описывают ранее принятые автоматические gates. TASK 06 должен повторно
проверить их совместно на точном release-candidate SHA.

## Автоматические gates TASK 06

Итоговые значения записываются только после фактического прогона. Переносить
результаты с другой ветки или SHA запрещено.

| Gate | Требование | Источник доказательства |
|---|---|---|
| Version consistency | `pyproject.toml`, runtime `VERSION` и provenance = `0.5.0rc1` | version tests / package metadata |
| Dependency health | `pip check` без ошибок | локальный лог и CI |
| Static checks | Ruff и `compileall` успешно | локальный лог и CI |
| Full tests | полный pytest успешно | JUnit / CI summary |
| Core coverage | не ниже 80% | coverage XML / CI summary |
| Application | Streamlit AppTest успешно | pytest / CI |
| Production demo | CLI demo успешно | demo artifact |
| Semantic integrity | verifier принимает полный пакет | verifier log |
| Acceptance framework | все обязательные synthetic cases соответствуют manifest | acceptance reports |
| CI matrix | Windows/Ubuntu × Python 3.10/3.11/3.12 и Required CI зелёные | GitHub Actions run exact head |

### Фактический локальный результат от 2026-07-12

На объединённом рабочем дереве TASK 06 получены следующие результаты:

| Gate | Результат |
|---|---|
| Version consistency | PASS — package/runtime/provenance `0.5.0rc1` |
| Dependency health | PASS — `pip check` без ошибок |
| Static checks | PASS — Ruff; `compileall` также включает каталог `acceptance` |
| Full tests | PASS — 415 tests |
| Core coverage | PASS — 130 tests, 84.57% |
| Application | PASS — 8 Streamlit AppTest tests внутри full pytest |
| Production demo | PASS — создан полный `reproducibility.zip` |
| Semantic integrity | PASS — demo artifact verifier |
| Acceptance framework | PASS — exit 0, 10/10 synthetic cases, 0 critical mismatches |
| Engineering acceptance | **NOT GRANTED** — `engineering_acceptance=false`, 10 gates unsigned |
| CI matrix exact remote head | PENDING до создания commit и Draft PR; результаты другой ветки не переносятся |

Локальные JSON/Markdown/HTML acceptance reports детерминированы и содержат expected
и actual для каждой проверки. Три real-case templates остаются `unsigned`; локальный
PASS не заменяет инженерную подпись.

Команда приёмочного framework:

```powershell
soil-stamp acceptance-run acceptance/manifest.json --out acceptance/results
```

Ожидаемые представления одного результата:

- `acceptance/results/acceptance_report.json` — машинный отчёт;
- `acceptance/results/acceptance_report.md` — проверяемое текстовое представление;
- `acceptance/results/acceptance_report.html` — автономное представление для просмотра.

Критическое несовпадение golden values или допусков, scientific flags,
failure/censoring, профиля/диапазона/статуса `E`, хэшей или состава report package
должно давать ненулевой exit code.

## Engineering gates — требуют подписи

Следующие gates не могут стать `approved` только от автоматического CI:

- минимум три реальных испытания заполнены без изменения исходного raw;
- для каждого реального испытания проверены паспорт, единицы, геометрия штампа,
  последовательность ступеней и классификация событий;
- подтверждены паспорта, поверка, назначение и координаты реальных индикаторов;
- подтверждены диапазон давления, профиль и статус каждого основного `E`;
- проверены реальные `pair_id`, группы и решения о публикационной кривой;
- проверены интервалы failure/right-censoring и отсутствие выдуманной осадки;
- независимый расчёт сопоставлен с результатами программы и приложена ссылка;
- указаны reviewer, дата, решение и подпись инженера для каждого real case;
- инженер заполнил и проверил библиографическую трассируемость профиля
  `antonov_round_stamp_v1`;
- ручной ввод реального протокола и перенос raw независимо проверены инженером;
- SQLite project archive ещё не реализован; до его эксплуатации требуются
  реализация, проверка migrations/immutable revisions/backup/restore и подпись;
- clean-machine Windows portable не входит в RC и требует отдельного smoke test;
- владелец принял отдельное решение о лицензии и допустимости распространения.

До выполнения этих пунктов реальные cases обязаны сохранять
`signoff_status=unsigned`; `reviewer` нельзя подменять именем автора программы или
автоматического агента.

## Решение о выпуске

Зелёный CI делает commit техническим кандидатом на инженерную приёмку. Он не
разрешает автоматически merge, tag, публикацию final release или эксплуатацию.
Такие действия выполняются владельцем отдельно после инженерных подписей и
разрешения лицензионного gate.
