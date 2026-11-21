@echo off
REM TerraScope - start the app locally (PC or demo laptop). Double-click.
REM Uses .venv (PC, made by setup.bat) if present, else .venv-demo; creates .venv-demo on first run
REM from the bundled wheels\ folder (no internet needed) or, if there is none, from the internet.
REM The server listens on 127.0.0.1 only; the browser opens once /api/health answers.
REM pip + server output also go to logs\start_local_<YYYYmmdd>.txt; short history in logs\bat_history.txt
setlocal EnableExtensions
cd /d "%~dp0"
title TerraScope - server
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "RC=0"
if not exist "logs\" mkdir "logs"
set "BATLOG=logs\bat_history.txt"
if not defined TERRASCOPE_PORT set "TERRASCOPE_PORT=8000"
set "URL=http://127.0.0.1:%TERRASCOPE_PORT%"
call :note "start_local.bat started in %CD%"

REM ---------------------------------------------------------------- pick the environment
set "VPY=.venv\Scripts\python.exe"
if exist "%VPY%" goto :pc_env
set "VPY=.venv-demo\Scripts\python.exe"
if exist "%VPY%" goto :demo_env
goto :create_demo

:pc_env
call :note "using .venv (PC environment)"
"%VPY%" -c "import torch, fastapi, uvicorn, numpy, scipy, matplotlib, PIL, multipart" >nul 2>&1
if errorlevel 1 goto :pc_broken
goto :checks

:demo_env
call :note "using .venv-demo"
"%VPY%" -c "import torch, fastapi, uvicorn, numpy, scipy, matplotlib, PIL, multipart, onnxruntime" >nul 2>&1
if errorlevel 1 goto :install_demo
goto :demo_runtime

REM ---------------------------------------------------------------- first run: create .venv-demo
:create_demo
call :note "first run: creating .venv-demo (one time, a few minutes)"
set "WHEELPY="
if exist "wheels\PYTHON_VERSION.txt" set /p WHEELPY=<"wheels\PYTHON_VERSION.txt"
set "PY="
if not defined WHEELPY goto :find_any_py
py -%WHEELPY% -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=py -%WHEELPY%"
if defined PY goto :make_venv
call :note "WARNING: the bundled wheels are for Python %WHEELPY%, which is not installed - will try another Python and the internet"
:find_any_py
for %%v in (3.12 3.13 3.11 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY=py -%%v"
  )
)
if defined PY goto :make_venv
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto :make_venv
call :note "ERROR: Python 3.10-3.13 not found."
echo   Install Python 3.12 (64-bit): run the installer in the python-installer folder if this bundle has one,
echo   or download it from https://www.python.org/downloads/windows/ - tick "Add python.exe to PATH".
goto :fail

:make_venv
call :note "creating .venv-demo with %PY%"
%PY% -m venv ".venv-demo"
if errorlevel 1 goto :fail
set "VPY=.venv-demo\Scripts\python.exe"

:install_demo
if not exist "requirements-demo.txt" goto :no_reqs
if not exist "wheels\" goto :install_online
call :note "installing packages from the bundled wheels folder (offline)"
"%VPY%" -m terrascope.utils.tee_run --job start_local -- "%VPY%" -m pip install --no-index --find-links "wheels" -r "requirements-demo.txt"
if not errorlevel 1 goto :installed
call :note "WARNING: offline install from wheels failed - trying the internet"
:install_online
call :note "installing packages from the internet (CPU PyTorch, about 1 GB download - estimate)"
"%VPY%" -m terrascope.utils.tee_run --job start_local -- "%VPY%" -m pip install -r "requirements-demo.txt" --extra-index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :install_fail
:installed
"%VPY%" -c "import torch, fastapi, uvicorn, numpy, scipy, matplotlib, PIL, multipart, onnxruntime" >nul 2>&1
if errorlevel 1 goto :install_fail
call :note "packages installed"

:demo_runtime
REM the demo laptop runs the foundation models from ONNX files exported on the PC
if not defined TERRASCOPE_RUNTIME if exist "artifacts\onnx\" set "TERRASCOPE_RUNTIME=onnx"
REM keep the analyses precomputed on the PC (cache\) from being evicted by the default 500 MB limit
if not defined TERRASCOPE_CACHE_MB set "TERRASCOPE_CACHE_MB=3000"
REM server binds 127.0.0.1 only, so the per-IP limit just gets in the way of a live demo
if not defined TERRASCOPE_RATE_LIMIT set "TERRASCOPE_RATE_LIMIT=0"

:checks
if not exist "artifacts\ckpt\finetuned.pt" if not exist "artifacts\ckpt\v1\finetuned.pt" goto :no_ckpt
REM already running? then just open the browser
"%VPY%" -m terrascope.utils.open_when_ready "%URL%" --timeout 1 >nul 2>&1
if not errorlevel 1 goto :already

echo.
echo   TerraScope starting at  %URL%
echo   The browser opens by itself when the server is ready. Close this window or press Ctrl+C to stop.
echo.
start "" /b "%VPY%" -m terrascope.utils.open_when_ready "%URL%" --timeout 600
call :note "server starting on %URL% (runtime=%TERRASCOPE_RUNTIME%)"
REM server output is shown here AND appended to logs\start_local_<date>.txt
"%VPY%" -m terrascope.utils.tee_run --job start_local -- "%VPY%" -m uvicorn terrascope.server.app:app --host 127.0.0.1 --port %TERRASCOPE_PORT%
if errorlevel 1 goto :server_fail
goto :end

:already
call :note "server already running at %URL% - opened the browser"
goto :end

:pc_broken
call :note "ERROR: .venv is incomplete - double-click setup.bat"
goto :fail

:no_reqs
call :note "ERROR: requirements-demo.txt missing - this folder is incomplete"
goto :fail

:install_fail
call :note "ERROR: installing packages failed. Delete the .venv-demo folder and try again with internet, or rebuild the bundle with wheels."
goto :fail

:no_ckpt
call :note "ERROR: artifacts\ckpt\finetuned.pt missing - this folder is incomplete"
goto :fail

:server_fail
call :note "the server stopped with an error code. If you pressed Ctrl+C this is normal. Otherwise: is port %TERRASCOPE_PORT% in use by another program? Read the messages above."
goto :fail

:fail
set "RC=1"

:end
call :note "start_local.bat finished with exit code %RC%"
echo.
pause
exit /b %RC%

:note
REM delayed expansion prints the text literally, even with & ( ) ^ in folder names
set "NOTE=%~1"
setlocal EnableDelayedExpansion
echo [start_local] !NOTE!
>> "%BATLOG%" echo [%date% %time%] [start_local] !NOTE!
endlocal
exit /b 0
