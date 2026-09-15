@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto failed
python main.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Program failed. Check the error above.
pause
exit /b 1
