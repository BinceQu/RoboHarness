@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync.ps1" -Uninstall
pause
