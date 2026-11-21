@echo off
REM TerraScope - GPU / disk / RAM check -> artifacts\pc_info.json
REM Double-click. Writes logs\gpu_check_<date-time>.txt (details) and a line in logs\bat_history.txt.
setlocal EnableExtensions
cd /d "%~dp0"
title TerraScope - gpu_check
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "RC=0"
if not exist "logs\" mkdir "logs"
set "BATLOG=logs\bat_history.txt"
call :note "gpu_check.bat started"

set "VPY=.venv\Scripts\python.exe"
if exist "%VPY%" goto :run
REM before setup.bat: use any installed Python (shows driver / disk / RAM; torch will be reported missing)
set "VPY="
for /f "delims=" %%p in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "VPY=%%p"
if not defined VPY goto :novenv
call :note ".venv not found yet - using %VPY%"

:run
"%VPY%" "scripts\pc\gpu_check.py" --strict %*
if errorlevel 1 goto :fail
goto :ok

:ok
echo.
call :note "gpu_check OK - next: double-click get_data.bat"
goto :end

:novenv
call :note "ERROR: .venv not found - double-click setup.bat first"
goto :fail

:fail
set "RC=1"
echo.
call :note "gpu_check FAILED - read the JOB FAILED block above; full log: newest logs\gpu_check_*.txt"

:end
call :note "gpu_check.bat finished with exit code %RC%"
echo.
pause
exit /b %RC%

:note
REM delayed expansion prints the text literally, even with & ( ) ^ in folder names
set "NOTE=%~1"
setlocal EnableDelayedExpansion
echo [gpu_check] !NOTE!
>> "%BATLOG%" echo [%date% %time%] [gpu_check] !NOTE!
endlocal
exit /b 0
