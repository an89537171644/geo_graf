$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
& py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Требуется Python 3.10 или новее." }
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Не удалось создать .venv." }
}
& ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Окружение .venv создано старой версией Python. Удалите .venv и запустите снова."
}
if (-not (Test-Path -LiteralPath ".venv\.installed-0.4.1a2-indicators")) {
    & ".venv\Scripts\python.exe" -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить зависимости." }
    New-Item -ItemType File -Path ".venv\.installed-0.4.1a2-indicators" -Force | Out-Null
}
& ".venv\Scripts\python.exe" -m streamlit run app.py
if ($LASTEXITCODE -ne 0) { throw "Streamlit завершился с ошибкой." }
