@echo off
REM TerraScope - evaluate trained models on the real test splits -> artifacts\metrics_real.json
REM Double-click. Writes logs\evaluate_<date-time>.txt (details) and a line in logs\bat_history.txt.
setlocal EnableExtensions
cd /d "%~dp0"
title TerraScope - evaluate
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "RC=0"
if not exist "logs\" mkdir "logs"
set "BATLOG=logs\bat_history.txt"
call :note "evaluate.bat started"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" goto :novenv
"%VPY%" "scripts\pc\evaluate_real.py" %*
if errorlevel 1 goto :fail
goto :ok

:ok
echo.
call :note "evaluate OK - results in artifacts\metrics_real.json; next: test_all.bat"
goto :end

:novenv
call :note "ERROR: .venv not found - double-click setup.bat first"
goto :fail

:fail
set "RC=1"
echo.
call :note "evaluate FAILED - read the JOB FAILED block above; full log: newest logs\evaluate_*.txt"

:end
call :note "evaluate.bat finished with exit code %RC%"
echo.
pause
exit /b %RC%

:note
REM delayed expansion prints the text literally, even with & ( ) ^ in folder names
set "NOTE=%~1"
setlocal EnableDelayedExpansion
echo [evaluate] !NOTE!
>> "%BATLOG%" echo [%date% %time%] [evaluate] !NOTE!
endlocal
exit /b 0
