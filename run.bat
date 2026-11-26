@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtualenv missing. Run build.bat first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -X utf8 -m app.gradio_app %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
exit /b %RC%
