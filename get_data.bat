@echo off
REM TerraScope - download real datasets / model weights into data\real
REM Double-click. Writes logs\get_data_<date-time>.txt (details) and a line in logs\bat_history.txt.
setlocal EnableExtensions
cd /d "%~dp0"
title TerraScope - get_data
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "RC=0"
if not exist "logs\" mkdir "logs"
set "BATLOG=logs\bat_history.txt"
REM Usage: get_data.bat [all|floods|cloud|burn|landcover|dem|weights] [extra get_data.py options]   (double-click = menu, default all)
set "TARGET=%~1"
REM extra options after the target are passed on to get_data.py (e.g. get_data.bat cloud --max-patches 500)
set "EXTRA="
:argloop
if "%~2"=="" goto :argdone
set EXTRA=%EXTRA% %2
shift
goto :argloop
:argdone
if not "%TARGET%"=="" goto :check_target
echo.
echo   What to download?  A=all except cloud (default in 30 s)  F=floods  C=cloud  B=burn  L=landcover  D=dem  W=weights
choice /c AFCBLDW /t 30 /d A /n /m "  Choice [A,F,C,B,L,D,W]: "
REM errorlevel 255 = choice could not read the console (no window): use the default
if errorlevel 255 set "TARGET=all"& goto :check_target
if errorlevel 7 set "TARGET=weights"& goto :check_target
if errorlevel 6 set "TARGET=dem"& goto :check_target
if errorlevel 5 set "TARGET=landcover"& goto :check_target
if errorlevel 4 set "TARGET=burn"& goto :check_target
if errorlevel 3 set "TARGET=cloud"& goto :check_target
if errorlevel 2 set "TARGET=floods"& goto :check_target
set "TARGET=all"

:check_target
for %%t in (all floods cloud burn landcover dem weights) do if /i "%TARGET%"=="%%t" goto :target_ok
call :note "ERROR: unknown target '%TARGET%' (use all, floods, cloud, burn, landcover, dem or weights)"
goto :fail

:target_ok
call :note "get_data.bat started, target=%TARGET%"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" goto :novenv
"%VPY%" "scripts\pc\get_data.py" %TARGET% %EXTRA%
if errorlevel 1 goto :fail
goto :ok

:ok
echo.
call :note "get_data OK - next: double-click train.bat"
goto :end

:novenv
call :note "ERROR: .venv not found - double-click setup.bat first"
goto :fail

:fail
set "RC=1"
echo.
call :note "get_data FAILED - read the JOB FAILED block above; full log: newest logs\get_data_*.txt"

:end
call :note "get_data.bat finished with exit code %RC%"
echo.
pause
exit /b %RC%

:note
REM delayed expansion prints the text literally, even with & ( ) ^ in folder names
set "NOTE=%~1"
setlocal EnableDelayedExpansion
echo [get_data] !NOTE!
>> "%BATLOG%" echo [%date% %time%] [get_data] !NOTE!
endlocal
exit /b 0
