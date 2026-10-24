@echo off
REM ACCS 启动（源码运行）
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo 未找到 .venv，请先双击 setup.bat 安装环境
  pause
  exit /b 1
)
echo ACCS 正在启动：http://127.0.0.1:8000
.venv\Scripts\python.exe -m accs.web --port 8000
pause
