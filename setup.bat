@echo off
REM TerraScope - PC setup (Windows 11, NVIDIA RTX 50xx). Double-click. Safe to run again.
REM Creates .venv with Python 3.12 (or 3.13), installs CUDA 12.8 PyTorch + requirements-pc.txt,
REM checks the GPU, and makes a first git commit if git is installed and this is not a git repo yet.
REM Detailed log: logs\setup_<date-time>.txt   Short history of every .bat run: logs\bat_history.txt
setlocal EnableExtensions
cd /d "%~dp0"
title TerraScope - setup
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "RC=0"
if not exist "logs\" mkdir "logs"
set "BATLOG=logs\bat_history.txt"
call :note "setup.bat started in %CD%"

REM ---------------------------------------------------------------- find Python 3.12 / 3.13 (64-bit)
REM terratorch 1.2.13 needs torchgeo 0.9 which requires Python 3.12+; 3.12 has every Windows wheel.
set "PY="
py -3.12 -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 set "PY=py -3.12"
if defined PY goto :have_py
py -3.13 -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 set "PY=py -3.13"
if defined PY goto :have_py
python -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,12),(3,13)) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto :have_py
call :note "ERROR: Python 3.12 or 3.13 (64-bit) not found."
echo.
echo   Install Python 3.12 (64-bit) from https://www.python.org/downloads/windows/
echo   - tick "Add python.exe to PATH" and keep "py launcher" selected,
echo   - or run in a terminal:  winget install Python.Python.3.12
echo   Python 3.10 / 3.11 cannot install terratorch 1.2.13 (needs torchgeo 0.9, Python 3.12+).
goto :fail

:have_py
for /f "delims=" %%v in ('%PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%v"
call :note "using %PY% (Python %PYVER%)"

REM ---------------------------------------------------------------- virtual environment
set "VPY=.venv\Scripts\python.exe"
if exist "%VPY%" goto :check_venv
call :note "creating .venv ..."
%PY% -m venv ".venv"
if errorlevel 1 goto :venv_fail
goto :install

:check_venv
"%VPY%" -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,12),(3,13)) else 1)" >nul 2>&1
if errorlevel 1 goto :venv_wrong
call :note ".venv already exists - reusing it"

:install
REM ---------------------------------------------------------------- packages (logged by setup_env.py)
call :note "installing packages - first run downloads several GB, see logs\setup_*.txt"
"%VPY%" "scripts\pc\setup_env.py" %*
if errorlevel 2 goto :gpu_warn
if errorlevel 1 goto :fail
goto :git

:gpu_warn
call :note "WARNING: packages installed but PyTorch cannot use the GPU - read the JOB FAILED block above"
set "RC=2"

:git
REM ---------------------------------------------------------------- git (optional)
where git >nul 2>&1
if errorlevel 1 goto :nogit
if exist ".git\" goto :gitexists
call :note "git: creating repository and first commit"
git init -q
if errorlevel 1 goto :gitfail
git add -A
if errorlevel 1 goto :gitfail
set "GITID="
git config user.email >nul 2>&1
if errorlevel 1 set "GITID=-c user.name=TerraScope -c user.email=terrascope@localhost"
git %GITID% commit -q -m "TerraScope: initial commit (setup.bat)"
if errorlevel 1 goto :gitfail
call :note "git: first commit done"
goto :done
:nogit
call :note "git not installed - skipped (optional)"
goto :done
:gitexists
call :note "git repository already exists - no commit made"
goto :done
:gitfail
call :note "WARNING: git step failed (optional, setup itself is fine)"
goto :done

:venv_fail
call :note "ERROR: could not create .venv with %PY%"
goto :fail

:venv_wrong
REM an older .venv (e.g. Python 3.14 from the first start_local.bat) is moved aside, not deleted
set "OLDVENV=.venv_old_%RANDOM%"
call :note "existing .venv uses an unsupported Python - moving it to %OLDVENV% and creating a new one"
ren ".venv" "%OLDVENV%"
if errorlevel 1 goto :venv_move_fail
call :note "creating .venv ..."
%PY% -m venv ".venv"
if errorlevel 1 goto :venv_fail
goto :install

:venv_move_fail
call :note "ERROR: could not rename the old .venv folder (is the app still running? close its window)."
echo   Close any TerraScope window, then double-click setup.bat again.
goto :fail

:done
echo.
if "%RC%"=="0" call :note "SETUP OK - next: double-click gpu_check.bat"
goto :end

:fail
set "RC=1"
echo.
call :note "SETUP FAILED - read the message above; full log in the logs folder (newest setup_*.txt)"

:end
call :note "setup.bat finished with exit code %RC%"
echo.
pause
exit /b %RC%

:note
REM delayed expansion prints the text literally, even with & ( ) ^ in folder names
set "NOTE=%~1"
setlocal EnableDelayedExpansion
echo [setup] !NOTE!
>> "%BATLOG%" echo [%date% %time%] [setup] !NOTE!
endlocal
exit /b 0
