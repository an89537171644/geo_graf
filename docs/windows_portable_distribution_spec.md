# Техническое задание: Windows portable distribution

## Статус документа

**Draft technical assignment.** Документ задаёт отдельную будущую работу и не
реализует portable package. Он не разрешает распространение, не создаёт final tag,
не объявляет программу окончательным релизом и не закрывает инженерные gates.

Текущие `run_app.bat` и `run_app.ps1` не являются portable distribution: им нужны
установленный Python и сетевой `pip install`. Их успешный запуск нельзя использовать
как доказательство выполнения этого ТЗ.

## 1. Цель

Подготовить самодостаточный ZIP для локального запуска Soil Stamp в Windows без
системного Python, Git, прав администратора и доступа в интернет. Интерфейс
Streamlit открывается в системном браузере, а вычисления и пользовательские данные
остаются на локальном компьютере.

Результат успешной технической проверки означает только пригодность конкретного
prerelease-артефакта к дальнейшему инженерному рассмотрению.

## 2. Первая поддерживаемая конфигурация

- Windows 11 x64 на servicing release, поддерживаемом производителем на дату сборки;
- встроенный CPython 3.12.x x64;
- стандартная учётная запись без elevation;
- offline runtime после получения и проверки ZIP.

Windows x86, ARM64, Server, Wine и Windows 10 не входят в первую конфигурацию и
требуют отдельного решения о поддержке и отдельной приёмочной матрицы. Репозиторный
CI Python 3.10–3.12 остаётся проверкой исходного проекта, но portable artifact
содержит ровно один зафиксированный runtime.

## 3. Packaging ADR и технический spike

Базовый вариант для spike: PyInstaller `--onedir`, упакованный затем в ZIP.
`--onefile` не является базовым вариантом из-за непрозрачной временной распаковки,
сложной диагностики и повышенного риска ложных срабатываний защиты. Если spike
покажет неподдерживаемые imports/resources Streamlit, допускается документированный
fallback на CPython embeddable x64 с заранее подготовленным `site-packages`.

До реализации должен быть принят ADR, содержащий:

- выбранный bundler и его точную версию;
- список hidden imports, data files и исключённых dev/test-компонентов;
- способ включения `app.py`, `soilstamp`, шрифтов/Matplotlib и Streamlit assets;
- доказательство, что source и packaged pipeline дают одинаковые научные результаты;
- причины отклонения от базового варианта, если выбран fallback.

MSI/MSIX, installer, auto-update, bundled Chromium/WebView и сетевой backend не
входят в это ТЗ.

## 4. Состав portable-каталога

После распаковки пользователь должен видеть как минимум:

- `SoilStamp.exe` — launcher без shell-install команд;
- каталог неизменяемых runtime-файлов;
- `README_PORTABLE.md` с запуском, остановкой и диагностикой;
- `build-info.json`;
- `sbom.cdx.json` или эквивалентный SPDX SBOM;
- `THIRD_PARTY_NOTICES.txt`;
- файл проверки цифровой подписи;
- локальный UTF-8 log-каталог либо явную настройку его размещения.

Каталог программы считается read-only. Пользовательские raw и результаты туда не
записываются.

## 5. Launcher

Launcher обязан:

1. запускаться без Python/Git в `PATH`, elevation, установки пакетов и изменения
   реестра/переменных окружения;
2. слушать только loopback `127.0.0.1`, никогда `0.0.0.0`;
3. отключать Streamlit telemetry и внешние сетевые обращения;
4. выбирать свободный локальный порт с ограниченным retry и обрабатывать конфликт;
5. дождаться health check перед открытием системного браузера;
6. явно управлять повторным запуском: single instance либо независимые экземпляры
   с разными портами и каталогами;
7. корректно завершать дочерний процесс и удалять только собственные временные файлы;
8. писать понятный UTF-8 diagnostic log без raw-таблиц, секретов и персональных
   данных;
9. сообщать версию, source commit и путь к log при ошибке запуска.

Firewall rule, Windows service и внешнее прослушивание портов не создаются.

## 6. Входы, выходы и сохранность данных

Portable artifact должен обрабатывать тем же production pipeline:

- CSV/TXT и XLSX/XLSM как данные без исполнения макросов;
- metadata JSON и manual-entry draft JSON;
- выходные CSV/JSON, SVG/PDF/PNG, `report.html`, `report.xlsx`,
  `artifact_manifest.json`, `approval_report.zip` и `reproducibility.zip`.

Требования:

- точные raw bytes и их SHA-256 сохраняются без скрытых исправлений;
- результаты пишутся только в выбранный пользователем writable-каталог;
- поддерживаются пробелы, кириллица и согласованный набор длинных путей;
- существующие ограничения XLSX/ZIP/XML, safe archive paths, formula-safe XLSX и
  запрет исполняемого/external content не ослабляются;
- diagnostic/evidence bundle не включает реальные raw без явного действия
  пользователя;
- ни один технический PASS не меняет real-case `signoff_status` автоматически.

## 7. Supply chain и воспроизводимость

Перед сборкой нужны:

- полный Windows lock всех runtime transitive dependencies с hashes;
- offline wheelhouse, проверяемый установкой с `--require-hashes`;
- зафиксированные CPython, bundler и build tools;
- SBOM, перечень third-party licenses/assets и vulnerability scan;
- отдельное решение владельца о лицензии и допустимости распространения;
- запрет включения build secrets и лишних test/dev packages в пользовательский ZIP.

`build-info.json` должен содержать product/version и prerelease status, source commit
и tree hash, SHA-256 dependency lock, версии CPython/bundler/runner, UTC build time,
workflow run и SHA-256 SBOM. Git не требуется на пользовательском компьютере:
provenance встраивается на этапе сборки.

Две чистые unsigned-сборки одного source SHA должны давать одинаковый content
manifest. Authenticode-подпись и timestamp проверяются отдельным слоем, поскольку
они могут изменять bytes подписанного бинарного файла.

## 8. Security и подпись

- EXE/DLL подписываются Authenticode; certificate thumbprint и timestamp
  фиксируются в evidence.
- Ключ и пароль не хранятся в репозитории, ZIP, workflow artifacts или logs;
  используется защищённая signing environment либо аппаратно защищённый ключ.
- После упаковки выполняются signature verification, Microsoft Defender и
  согласованный корпоративный AV scan.
- Во время offline smoke test контролируется отсутствие соединений за пределами
  loopback.
- Приложение не требует elevation и не запускает пользовательские строки как shell,
  Python или табличные формулы.

Неподписанная exploratory build может использоваться только внутри технического
spike и обязана иметь видимую маркировку; она не является распространяемым релизом.

## 9. Build/CI pipeline

Отдельный защищённый workflow должен:

1. checkout точного tag/commit без persistent credentials;
2. восстановить проверенный offline toolchain по hashes;
3. выполнить `pip check`, Ruff, `compileall`, полный pytest и core coverage ≥80%;
4. выполнить CLI demo, semantic verifier и synthetic `acceptance-run`;
5. собрать unsigned portable tree и сформировать content manifest;
6. выполнить packaged/source parity;
7. подписать разрешённый artifact отдельным защищённым job;
8. проверить подпись, SBOM, licenses и AV results;
9. опубликовать только полный evidence set с retention policy.

Обычный GitHub-hosted Windows runner, на котором уже установлен Python/Git, не
заменяет clean-machine test.

## 10. Clean-machine acceptance

Проверки выполняются на новой Windows VM/Sandbox без Python, Git и admin-прав, с
отключённым внешним интернетом:

- [ ] SHA-256 ZIP и цифровая подпись проверены до распаковки;
- [ ] запуск и остановка проходят offline;
- [ ] каталог программы read-only, output расположен отдельно;
- [ ] путь содержит пробелы и кириллицу; согласованный long-path case проходит;
- [ ] занятый default port обрабатывается без внешнего bind;
- [ ] повторный запуск и аварийное/штатное закрытие не оставляют чужих процессов;
- [ ] интерфейс проверен при DPI 100%, 150% и 200%;
- [ ] обработаны CSV, strict XLSX, interactive mapping и manual draft;
- [ ] CLI demo и semantic verifier прошли;
- [ ] synthetic `acceptance-run`: 10/10, 0 critical,
  `engineering_acceptance=false`;
- [ ] результаты source и packaged pipeline совпали на точном source SHA;
- [ ] raw SHA и report-package manifest прошли независимую проверку;
- [ ] malformed/oversized XLSX, ZIP traversal, formula и external-link cases
  отклонены ожидаемым образом;
- [ ] внешних сетевых соединений нет;
- [ ] SBOM, license inventory, signature и AV evidence приложены.

Три реальных испытания остаются `unsigned`, пока их отдельно не подпишет инженер;
clean-machine smoke test не заменяет эту проверку.

## 11. Выходные артефакты будущей задачи

- `soil-stamp-antonov-<prerelease>-windows-x86_64-portable.zip`;
- `SHA256SUMS.txt`;
- `build-info.json`;
- `sbom.cdx.json`;
- `THIRD_PARTY_NOTICES.txt`;
- `signature-verification.txt`;
- validation evidence ZIP с JUnit, coverage, acceptance JSON/Markdown/HTML,
  demo/verifier/build/security logs без реальных raw.

## 12. Явные non-goals

- SQLite, migrations, immutable/approved revisions, backup/restore и project archive;
- изменение научных формул, расчётного ядра, графиков Антонова или fixtures;
- автоматическое утверждение real cases;
- remote server, cloud transfer, auto-update и installer;
- final tag, merge или заявление окончательного релиза.

## 13. Неподписанные gates portable-задачи

- packaging ADR и Windows support matrix;
- полный hash-locked dependency set и offline wheelhouse;
- dependency/license/assets review и решение владельца о распространении;
- signing identity/certificate и защищённый signing workflow;
- clean-machine functional/security tests;
- Defender/AV review;
- packaged/source scientific parity;
- portable provenance и воспроизводимый content manifest;
- инженерная usability-проверка;
- три real cases и методическая трассируемость `antonov_round_stamp_v1`.

SQLite остаётся отдельной будущей задачей и не реализуется в portable PR. Закрытие
перечисленных технических gates не присваивает программе статус final release.
