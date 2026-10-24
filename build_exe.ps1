# Build Windows executable (onedir) for ACCS
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File build_exe.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'No .venv found. Run setup.bat first.' }

& $py -m pip install --quiet pyinstaller

$pyArgs = @(
  '--noconfirm', '--clean',
  '--name', 'ACCS',
  '--console',
  '--add-data', 'accs/static;accs/static',
  '--collect-all', 'sherpa_onnx',
  '--collect-all', 'onnxruntime',
  '--collect-data', 'faster_whisper',
  '--collect-data', 'soundfile',
  '--collect-submodules', 'uvicorn',
  '--hidden-import', 'uvicorn.logging',
  '--hidden-import', 'uvicorn.loops.auto',
  '--hidden-import', 'uvicorn.protocols.http.auto',
  '--hidden-import', 'uvicorn.protocols.websockets.auto',
  '--hidden-import', 'uvicorn.lifespan.on',
  '--hidden-import', 'soundfile',
  '--exclude-module', 'torch',
  '--exclude-module', 'whisper',
  '--exclude-module', 'funasr',
  '--exclude-module', 'matplotlib',
  'run_web.py'
)
& $py -m PyInstaller @pyArgs

$dist = Join-Path $PSScriptRoot 'dist\ACCS'
Copy-Item 'config.yaml' $dist -Force
Copy-Item 'README.md' $dist -Force -ErrorAction SilentlyContinue

$bat = @"
@echo off
cd /d %~dp0
echo ACCS starting: http://127.0.0.1:8000
ACCS.exe --port 8000
pause
"@
Set-Content -Path (Join-Path $dist 'start.bat') -Value $bat -Encoding ASCII

$modelsLink = Join-Path $dist 'models'
if (-not (Test-Path -LiteralPath $modelsLink)) {
  try {
    New-Item -ItemType Junction -Path $modelsLink -Target (Join-Path $PSScriptRoot 'models') | Out-Null
  } catch {
    Write-Warning 'Could not create models junction; copy models/ next to ACCS.exe manually.'
  }
}

Write-Host "Build finished: $dist\ACCS.exe"
