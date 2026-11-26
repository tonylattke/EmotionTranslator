@echo off
setlocal
cd /d "%~dp0"

set "PYTHON="
for %%V in (3.11 3.10 3.12 3.9) do (
  if not defined PYTHON (
    py -%%V -c "import sys" >nul 2>&1 && set "PYTHON=py -%%V"
  )
)
if not defined PYTHON (
  python -c "import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1 && set "PYTHON=python"
)
if not defined PYTHON (
  echo ERROR: Python 3.9-3.12 not found. Install it with: winget install Python.Python.3.11
  pause
  exit /b 1
)

%PYTHON% "%~dp0scripts\setup_env.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
exit /b %RC%
