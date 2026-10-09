@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync.ps1" %*
if errorlevel 1 echo Sync failed. The existing offline page is still available.
pause
