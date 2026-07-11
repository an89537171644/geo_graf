@echo off
setlocal
cd /d "%~dp0"
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo Требуется Python 3.10 или новее.
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo Окружение .venv создано старой версией Python. Удалите .venv и запустите снова.
  exit /b 1
)
if not exist ".venv\.installed-0.5.0b2.dev1-manual" (
  ".venv\Scripts\python.exe" -m pip install -e .
  if errorlevel 1 exit /b 1
  type nul > ".venv\.installed-0.5.0b2.dev1-manual"
)
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 exit /b 1
