@echo off
REM Cortex CLI for Windows:  .\cx gui   |   .\cx "a question"   |   .\cx voice
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m cortex.cli.cx_app %*
) else (
  py -m cortex.cli.cx_app %*
)
