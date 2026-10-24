@echo off
REM ACCS 一键安装（源码运行）
cd /d %~dp0
echo === ACCS 环境安装 ===
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3.9) else (set PY=python)

%PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install python-docx
echo.
echo 安装完成。双击 启动.bat 运行；浏览器打开 http://127.0.0.1:8000
pause
