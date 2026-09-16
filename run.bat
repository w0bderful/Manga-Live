@echo off
chcp 65001 >nul
setlocal
set "failure_message=최소화된 실행 창을 열지 못했습니다. Windows 명령 프롬프트 실행 권한을 확인하세요."
if /i "%~1"=="--minimized" goto run
start "Manga Live" /min "%ComSpec%" /d /c ""%~f0" --minimized"
if errorlevel 1 goto failed
exit /b 0
:run
set "failure_message=프로젝트 폴더로 이동하지 못했습니다. 폴더 위치와 접근 권한을 확인하세요."
cd /d "%~dp0"
if errorlevel 1 goto failed
if not exist ".venv\Scripts\python.exe" (
    set "failure_message=가상환경이 없습니다. 먼저 setup.bat을 실행하세요."
    goto failed
)
set "failure_message=가상환경을 활성화하지 못했습니다. setup.bat을 다시 실행하세요."
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto failed
set "failure_message=프로그램 실행 중 오류가 발생했습니다. 위의 오류 내용과 logs\manga-live-YYYY-MM-DD.log를 확인하세요."
python main.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo 실행 실패: %failure_message%
echo 아무 키나 누르면 종료합니다.
pause >nul
exit /b 1
