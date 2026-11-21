@echo off
REM TerraScope - CPU speed benchmark per model -> artifacts\bench_laptop.json (run it on the demo laptop)
REM Double-click. Writes logs\bench_laptop_<date-time>.txt (details) and a line in logs\bat_history.txt.
setlocal EnableExtensions
cd /d "%~dp0"
title TerraScope - bench_laptop
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "RC=0"
if not exist "logs\" mkdir "logs"
set "BATLOG=logs\bat_history.txt"
call :note "bench_laptop.bat started"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=.venv-demo\Scripts\python.exe"
if not exist "%VPY%" goto :novenv
"%VPY%" "scripts\pc\bench_laptop.py" %*
if errorlevel 1 goto :fail
goto :ok

:ok
echo.
call :note "bench_laptop OK - results in artifacts\bench_laptop.json"
goto :end

:novenv
call :note "ERROR: .venv not found - double-click setup.bat first"
goto :fail

:fail
set "RC=1"
echo.
call :note "bench_laptop FAILED - read the JOB FAILED block above; full log: newest logs\bench_laptop_*.txt"

:end
call :note "bench_laptop.bat finished with exit code %RC%"
echo.
pause
exit /b %RC%

:note
REM delayed expansion prints the text literally, even with & ( ) ^ in folder names
set "NOTE=%~1"
setlocal EnableDelayedExpansion
echo [bench_laptop] !NOTE!
>> "%BATLOG%" echo [%date% %time%] [bench_laptop] !NOTE!
endlocal
exit /b 0
