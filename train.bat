@echo off
REM TerraScope - train / fine-tune the models (GPU)
REM Double-click. Writes logs\train_<date-time>.txt (details) and a line in logs\bat_history.txt.
setlocal EnableExtensions
cd /d "%~dp0"
title TerraScope - train
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "RC=0"
if not exist "logs\" mkdir "logs"
set "BATLOG=logs\bat_history.txt"
REM Usage: train.bat [quick|full|sim|paper] [extra train_all.py options]   (double-click = menu, default quick)
REM   quick = short end-to-end run to prove the pipeline works
REM   full  = the real model x task x seed grid (long)
REM   sim   = retrain the simulated-data TerraFM only
REM   paper = real-data runs for the revised paper (3 seeds x 6 models x flood + burn); see paper_experiments.bat
set "MODE=%~1"
REM extra options after the mode are passed on to train_all.py (e.g. train.bat full --models terrafm)
set "EXTRA="
:argloop
if "%~2"=="" goto :argdone
set EXTRA=%EXTRA% %2
shift
goto :argloop
:argdone
if not "%MODE%"=="" goto :check_mode
echo.
echo   Training mode?  Q=quick (default in 30 s)  F=full (long)  S=sim  P=paper
choice /c QFSP /t 30 /d Q /n /m "  Choice [Q,F,S,P]: "
REM errorlevel 255 = choice could not read the console (no window): use the default
if errorlevel 255 set "MODE=quick"& goto :check_mode
if errorlevel 4 set "MODE=paper"& goto :check_mode
if errorlevel 3 set "MODE=sim"& goto :check_mode
if errorlevel 2 set "MODE=full"& goto :check_mode
set "MODE=quick"

:check_mode
for %%m in (quick full sim paper) do if /i "%MODE%"=="%%m" goto :mode_ok
call :note "ERROR: unknown mode '%MODE%' (use quick, full, sim or paper)"
goto :fail

:mode_ok
call :note "train.bat started, mode=%MODE%"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" goto :novenv
"%VPY%" "scripts\pc\train_all.py" %MODE% %EXTRA%
if errorlevel 1 goto :fail
goto :ok

:ok
echo.
call :note "train OK - next: double-click evaluate.bat"
goto :end

:novenv
call :note "ERROR: .venv not found - double-click setup.bat first"
goto :fail

:fail
set "RC=1"
echo.
call :note "train FAILED - read the JOB FAILED block above; full log: newest logs\train_*.txt"

:end
call :note "train.bat finished with exit code %RC%"
echo.
pause
exit /b %RC%

:note
REM delayed expansion prints the text literally, even with & ( ) ^ in folder names
set "NOTE=%~1"
setlocal EnableDelayedExpansion
echo [train] !NOTE!
>> "%BATLOG%" echo [%date% %time%] [train] !NOTE!
endlocal
exit /b 0
