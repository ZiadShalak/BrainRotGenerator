@echo off
REM Activate the virtual environment for this project and keep the window open
cd /d "%~dp0"

REM Check if activate script exists
if exist "%~dp0.venv\Scripts\activate.bat" (
    call "%~dp0.venv\Scripts\activate.bat"
    title AI Vision Environment
) else (
    echo ERROR: Could not find activate.bat in .venv\Scripts
)

REM Keep window open for commands
cmd /k
