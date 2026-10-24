@echo off
REM Cortex launcher for Windows. Double-click this file.
cd /d "%~dp0"
where py >nul 2>nul && (py run.py %* & goto :done)
where python >nul 2>nul && (python run.py %* & goto :done)
echo Python 3 not found. Install it from https://python.org (check "Add to PATH").
pause
:done
