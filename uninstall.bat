@echo off
REM Cortex uninstaller for Windows. Double-click this file.
cd /d "%~dp0"
where py >nul 2>nul && (py uninstall.py %* & goto :done)
where python >nul 2>nul && (python uninstall.py %* & goto :done)
echo Python 3 not found.
pause
:done
