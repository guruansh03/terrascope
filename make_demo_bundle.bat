@echo off
REM TerraScope - build the offline demo bundle -> dist\terrascope_demo.zip
REM Double-click. Writes logs\make_demo_bundle_<date-time>.txt (details) and a line in logs\bat_history.txt.
setlocal EnableExtensions
cd /d "%~dp0"
title TerraScope - make_demo_bundle
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "RC=0"
if not exist "logs\" mkdir "logs"
set "BATLOG=logs\bat_history.txt"
REM Usage: make_demo_bundle.bat [extra options for scripts\pc\make_demo_bundle.py]
REM Wheels + Python installer are ALWAYS included (laptop needs no internet) unless --no-wheels / --no-python is given.
REM Double-click = menu: W = include them (default), N = leave them out (smaller zip).
set "OPTS=%*"
if not "%~1"=="" goto :run
echo.
echo   Include Python package wheels so the laptop needs no internet?
echo   W=yes, wheels + Python installer (default in 30 s)   N=no, smaller zip (laptop needs internet once)
choice /c WN /t 30 /d W /n /m "  Choice [W,N]: "
REM errorlevel 255 = choice could not read the console (no window): use the default
if errorlevel 255 set "OPTS=" & goto :run
if errorlevel 2 set "OPTS=--no-wheels --no-python" & goto :run
set "OPTS="

:run
call :note "make_demo_bundle.bat started, options: %OPTS%"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" goto :novenv
"%VPY%" "scripts\pc\make_demo_bundle.py" %OPTS%
if errorlevel 1 goto :fail
goto :ok

:ok
echo.
call :note "make_demo_bundle OK - copy dist\terrascope_demo.zip to the laptop"
goto :end

:novenv
call :note "ERROR: .venv not found - double-click setup.bat first"
goto :fail

:fail
set "RC=1"
echo.
call :note "make_demo_bundle FAILED - read the JOB FAILED block above; full log: newest logs\make_demo_bundle_*.txt"

:end
call :note "make_demo_bundle.bat finished with exit code %RC%"
echo.
pause
exit /b %RC%

:note
REM delayed expansion prints the text literally, even with & ( ) ^ in folder names
set "NOTE=%~1"
setlocal EnableDelayedExpansion
echo [make_demo_bundle] !NOTE!
>> "%BATLOG%" echo [%date% %time%] [make_demo_bundle] !NOTE!
endlocal
exit /b 0
