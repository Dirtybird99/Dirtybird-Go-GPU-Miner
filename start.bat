@echo off
REM Dirtybird Go GPU Miner -- Windows launcher.
setlocal
cd /d "%~dp0"

set "BIN=go-gpu-miner.exe"
if not exist "%BIN%" (
    echo error: %BIN% not found. Run this next to the release binary.
    pause
    exit /b 1
)

if not exist config.json if exist config.example.json copy /y config.example.json config.json >nul
echo Configure config.json or pass -d and -w on the command line.
echo Starting miner ^(Ctrl-C to stop^)...
echo.
"%BIN%" %*
