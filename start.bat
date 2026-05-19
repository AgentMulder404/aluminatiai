@echo off
:: AluminatiAi GPU Energy Agent - Windows launcher
:: Usage:  start.bat alum_YOUR_KEY_HERE
:: Or set ALUMINATAI_API_KEY in your environment first, then just run start.bat

if "%1" neq "" (
    set ALUMINATAI_API_KEY=%1
)

if "%ALUMINATAI_API_KEY%"=="" (
    echo Error: API key not provided.
    echo Usage:  start.bat alum_YOUR_KEY_HERE
    echo Or set ALUMINATAI_API_KEY as an environment variable first.
    exit /b 1
)

echo Starting AluminatiAi GPU Energy Agent...
echo API Key: %ALUMINATAI_API_KEY:~0,9%...
echo Press Ctrl+C to stop.
echo.

python main.py
