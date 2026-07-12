# Release readiness — 0.5.0rc1

## Статус кандидата

`0.5.0rc1` — **candidate for engineering acceptance**. Документ не является
сертификатом, подписью инженера, разрешением на эксплуатацию или объявлением
окончательного релиза. Финальный head TASK 06:
`8a946352cd69fd23c00122bbb5aff4071c65793a`. Владелец объединил его в
`main` merge-коммитом `c0a8ef0ddec8bd98b94364179abf3cf8c897ab63`.
Текущая документальная финализация выполняется в ветке
`release/0.5.0rc2-finalization`, созданной от этого merge-коммита; она не меняет
версию пакета. Классификация остаётся: **candidate for engineering acceptance,
not a final release**.

## Зафиксированное доказательство TASK 06

[GitHub Actions run 29199046654](https://github.com/an89537171644/geo_graf/actions/runs/29199046654)
выполнен для точного финального RC head
`8a946352cd69fd23c00122bbb5aff4071c65793a`:

- Windows/Ubuntu × Python 3.10/3.11/3.12: **6/6 matrix jobs SUCCESS**;
- агрегирующий job `Required CI`: **SUCCESS**;
- опубликованы 6/6 комплектов matrix artifacts.

Это доказательство относится к merged TASK 06 RC. Изменения в текущем Draft PR
обязаны отдельно пройти CI на собственном точном head SHA; результаты run
29199046654 нельзя выдавать за проверку ещё не созданного finalization commit.

## Локальный gate finalization-ветки

На рабочем дереве `release/0.5.0rc2-finalization` от 2026-07-12 выполнены:

- `pip check`, Ruff, `compileall app.py soilstamp tests scripts acceptance` и
  `git diff --check`: PASS;
- полный pytest: **420 passed**;
- calculation core: **130 passed**, coverage **84.57%** при gate 80%;
- CLI demo: PASS; semantic verifier: PASS; SHA-256 `reproducibility.zip`:
  `4ea3595263907f078955a6481ba0a4d5e1f262880770c592e0241468b3adca75`;
- acceptance-run: exit `0`, **10/10 synthetic cases PASS**, 0 critical,
  `synthetic_acceptance_passed=true`, `engineering_acceptance=false`, 10 gates
  остаются `unsigned`.

SHA-256 finalization acceptance reports: JSON
`178d66908b923393f2858b34930bb1d906e92aa43f65cbe9b46d376e402eb4c6`, Markdown
`22b518a12a36e4979bbd0b5289a201f56c93c82e18f64eaf056b3adbcfad1fdc`, HTML
`ac23b834aa99802d2023eb3bd083c3465c0e3e37f968e48c1886051fa239ae1f`.

GitHub Actions для exact head finalization-ветки является отдельным обязательным
gate. Его фактический run и все job results фиксируются в Draft PR после публикации
commit, чтобы status-only commit не делал только что полученный CI устаревшим.

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
| CI matrix exact TASK 06 RC head | PASS — [run 29199046654](https://github.com/an89537171644/geo_graf/actions/runs/29199046654), 6/6 matrix jobs SUCCESS, `Required CI` SUCCESS |

Локальные JSON/Markdown/HTML acceptance reports детерминированы и содержат expected
и actual для каждой проверки. Три real-case templates остаются `unsigned`; локальный
PASS не заменяет инженерную подпись.

Пошаговый порядок подготовки каждой из трёх реальных проверок приведён в
[`acceptance/real_cases/README.md`](../acceptance/real_cases/README.md). Канонические
templates в этом кандидате остаются незаполненными и неисполняемыми.

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
  `antonov_round_stamp_v1` по отдельному checklist в карточке методики;
- ручной ввод реального протокола и перенос raw независимо проверены инженером;
- SQLite project archive ещё не реализован; до его эксплуатации требуются
  реализация, проверка migrations/immutable revisions/backup/restore и подпись;
- clean-machine Windows portable не входит в RC и требует отдельной реализации и
  приёмки по [`windows_portable_distribution_spec.md`](windows_portable_distribution_spec.md);
- владелец принял отдельное решение о лицензии и допустимости распространения.

До выполнения этих пунктов реальные cases обязаны сохранять
`signoff_status=unsigned`; `reviewer` нельзя подменять именем автора программы или
автоматического агента.

## Решение о выпуске

Зелёный CI делает commit техническим кандидатом на инженерную приёмку. Он не
разрешает автоматически merge, tag, публикацию final release или эксплуатацию.
Такие действия выполняются владельцем отдельно после инженерных подписей и
разрешения лицензионного gate.
