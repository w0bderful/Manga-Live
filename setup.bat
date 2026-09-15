@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto failed
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Starting Manga OCR Translator...
call "%~dp0run.bat"
exit /b %errorlevel%
:failed
echo Setup failed. Check Python installation and the error above.
pause
exit /b 1
