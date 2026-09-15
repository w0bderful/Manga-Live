@echo off
chcp 65001 >nul
setlocal
set "failure_message=프로젝트 폴더로 이동하지 못했습니다. 폴더 위치와 접근 권한을 확인하세요."
cd /d "%~dp0"
if errorlevel 1 goto failed
if not exist ".venv\Scripts\python.exe" (
    set "failure_message=가상환경을 만들지 못했습니다. Python이 설치되어 있고 PATH에 등록되어 있는지 확인하세요."
    python -m venv .venv
    if errorlevel 1 goto failed
)
set "failure_message=가상환경을 활성화하지 못했습니다. .venv 폴더와 activate.bat 파일을 확인하세요."
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto failed
set "failure_message=필수 패키지를 설치하지 못했습니다. 인터넷 연결과 Python 버전을 확인하세요."
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo 설치가 완료되었습니다. Manga Live를 실행합니다...
call "%~dp0run.bat"
exit /b %errorlevel%
:failed
echo 설치 실패: %failure_message%
echo 위에 표시된 오류 내용을 확인하세요.
echo 아무 키나 누르면 종료합니다.
pause >nul
exit /b 1
