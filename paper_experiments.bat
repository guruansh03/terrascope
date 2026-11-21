@echo off
REM TerraScope - all experiments for the revised paper (reviewer round 1). Double-click.
REM Runs on the RTX 5060: about 1.5-2 h (estimate). Safe to re-run: finished runs are skipped (resume).
REM Each step writes its own logs\<job>_<date-time>.txt; every step start/result goes to logs\bat_history.txt.
setlocal EnableExtensions
cd /d "%~dp0"
title TerraScope - paper experiments
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "RC=0"
if not exist "logs\" mkdir "logs"
set "BATLOG=logs\bat_history.txt"
call :note "paper_experiments.bat started"

set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" goto :novenv
if not exist "artifacts\ckpt\" mkdir "artifacts\ckpt"
REM --force only on the very first run (retrain the simulated models from scratch); later runs resume
set "FORCE=--force"
if exist "artifacts\ckpt\paper_sim_started.txt" set "FORCE="
> "artifacts\ckpt\paper_sim_started.txt" echo started

set "STEP=1/6 fetch ImageNet ResNet-18 weights for the U-Net baseline (get_data weights)"
call :note "[%STEP%]"
"%VPY%" "scripts\pc\get_data.py" weights
if errorlevel 1 goto :fail

set "STEP=2/6 simulated data: 3 seeds x (4 pretraining variants + scratch + U-Net)"
call :note "[%STEP%] %FORCE%"
"%VPY%" -m terrascope.core.train paper_sim --seeds 0,1,2 --device cuda %FORCE%
if errorlevel 1 goto :fail

set "STEP=3/6 simulated data: evaluation with seeds, CIs, ablations, 128 px baselines"
call :note "[%STEP%]"
"%VPY%" -m terrascope.core.evaluate --paper --seeds 0,1,2 --device cuda
if errorlevel 1 goto :fail

set "STEP=4/6 risk-index sensitivity"
call :note "[%STEP%]"
"%VPY%" -m terrascope.core.risk_sensitivity
if errorlevel 1 goto :fail

set "STEP=5/6 real data: 3 seeds x 6 models x flood + burn (train_all paper)"
call :note "[%STEP%]"
"%VPY%" "scripts\pc\train_all.py" paper
if errorlevel 1 goto :fail

set "STEP=6/6 real data: evaluation with seeds, bootstrap CIs, paired tests"
call :note "[%STEP%]"
"%VPY%" "scripts\pc\evaluate_real.py"
if errorlevel 1 goto :fail

echo.
call :note "PAPER EXPERIMENTS OK - results in artifacts\metrics_paper_sim.json, artifacts\metrics_real.json, artifacts\risk_sensitivity.json"
goto :end

:novenv
call :note "ERROR: .venv not found - double-click setup.bat first"
goto :fail

:fail
set "RC=1"
echo.
call :note "PAPER EXPERIMENTS FAILED at step %STEP% - read the JOB FAILED block above (newest file in logs). Re-run this file to resume; finished runs are skipped."

:end
call :note "paper_experiments.bat finished with exit code %RC%"
echo.
pause
exit /b %RC%

:note
REM delayed expansion prints the text literally, even with & ( ) ^ in folder names
set "NOTE=%~1"
setlocal EnableDelayedExpansion
echo [paper_experiments] !NOTE!
>> "%BATLOG%" echo [%date% %time%] [paper_experiments] !NOTE!
endlocal
exit /b 0
