@echo off
REM TerraScope - run every test suite + API smoke test (+ UI test if Node.js is installed)
REM Double-click. Writes logs\test_all_<date-time>.txt (details) and a line in logs\bat_history.txt.
setlocal EnableExtensions
cd /d "%~dp0"
title TerraScope - test_all
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "RC=0"
if not exist "logs\" mkdir "logs"
set "BATLOG=logs\bat_history.txt"
call :note "test_all.bat started"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=.venv-demo\Scripts\python.exe"
if not exist "%VPY%" goto :novenv
"%VPY%" "tests\test_all.py" %*
if errorlevel 1 goto :fail
goto :ok

:ok
echo.
call :note "test_all OK - summary in the newest logs\test_all_*.txt"
goto :end

:novenv
call :note "ERROR: .venv not found - double-click setup.bat first"
goto :fail

:fail
set "RC=1"
echo.
call :note "test_all FAILED - read the JOB FAILED block above; full log: newest logs\test_all_*.txt"

:end
call :note "test_all.bat finished with exit code %RC%"
echo.
pause
exit /b %RC%

:note
REM delayed expansion prints the text literally, even with & ( ) ^ in folder names
set "NOTE=%~1"
setlocal EnableDelayedExpansion
echo [test_all] !NOTE!
>> "%BATLOG%" echo [%date% %time%] [test_all] !NOTE!
endlocal
exit /b 0
