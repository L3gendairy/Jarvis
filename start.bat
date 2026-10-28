@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Jarvis

rem ---- 1. Find a compatible Python (3.10 - 3.12) ------------------------------------------
set "PYCMD="
for %%C in ("py -3.12" "py -3.11" "py -3.10" "python") do (
    if not defined PYCMD (
        %%~C -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul && set "PYCMD=%%~C"
    )
)
if not defined PYCMD goto :no_python

rem ---- 2. Remove broken or copied virtual environments -------------------------------------
set "DELFAIL="
call :check_venv ".venv"
call :check_venv "workspace\.venv"
if defined DELFAIL goto :fail

rem ---- 3. Create .venv if needed -----------------------------------------------------------
set "FRESH="
if exist ".venv\Scripts\python.exe" goto :have_venv
echo Creating virtual environment .venv using: %PYCMD%
%PYCMD% -m venv .venv
if errorlevel 1 goto :venv_failed
if not exist ".venv\Scripts\python.exe" goto :venv_failed
set "FRESH=1"
:have_venv
set "VPY=.venv\Scripts\python.exe"

rem ---- 4. Always make sure the dependencies are installed (quick when already satisfied) ----
set "PIPQ=-q"
if defined FRESH set "PIPQ="
if defined FRESH "%VPY%" -m pip install --upgrade pip --disable-pip-version-check
if defined FRESH echo Installing dependencies. This takes a few minutes the first time...
if not defined FRESH echo Checking dependencies...
"%VPY%" -m pip install %PIPQ% --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :pip_failed

rem ---- 5. Launch with the venv's own Python, never the system one --------------------------
"%VPY%" main.py %*
if errorlevel 1 goto :run_failed
endlocal
exit /b 0


rem =========================================================================================
rem check_venv <dir>: delete the environment if it cannot be used on this PC.
rem   broken = no python.exe, no pyvenv.cfg, pyvenv.cfg "home" Python missing here (copied
rem   from another PC), or its python fails to start / is not 3.10-3.12 / has no pip.
rem =========================================================================================
:check_venv
set "VDIR=%~1"
if not exist "%VDIR%" exit /b 0
set "BAD="
if not exist "%VDIR%\Scripts\python.exe" set "BAD=python.exe is missing"
if not defined BAD if not exist "%VDIR%\pyvenv.cfg" set "BAD=pyvenv.cfg is missing"
if not defined BAD call :cfg_home "%VDIR%\pyvenv.cfg"
if not defined BAD "%VDIR%\Scripts\python.exe" -c "import sys, pip; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul || set "BAD=its Python does not run, is not 3.10-3.12, or has no pip"
if not defined BAD exit /b 0
echo.
echo [!] %VDIR% is not usable on this PC: %BAD%.
echo     Deleting it, a fresh one will be created.
rmdir /s /q "%VDIR%" >nul 2>nul
if exist "%VDIR%" goto :check_venv_stuck
exit /b 0
:check_venv_stuck
echo [X] Could not delete %VDIR%. Close Jarvis and anything using it, then run start.bat again.
set "DELFAIL=1"
exit /b 0

rem cfg_home <pyvenv.cfg>: BAD is set if the base Python named by "home =" does not exist here.
:cfg_home
for /f "usebackq tokens=1,* delims==" %%A in ("%~1") do (
    if /i "%%A"=="home " for /f "tokens=*" %%H in ("%%B") do if not exist "%%H\python.exe" set "BAD=the Python it was built from (%%H) is not installed on this PC"
)
exit /b 0


rem ---- Failure exits: every one pauses so the message can be read --------------------------
:no_python
echo.
echo [X] No compatible Python found. Jarvis needs Python 3.10, 3.11 or 3.12.
echo     Download Python 3.12 from:  https://www.python.org/downloads/windows/
echo     In the installer tick "Add python.exe to PATH", then run start.bat again.
goto :fail

:venv_failed
echo.
echo [X] Could not create the virtual environment .venv.
goto :fail

:pip_failed
echo.
echo [X] Installing requirements.txt failed. Check your internet connection and the messages above.
goto :fail

:run_failed
echo.
echo [X] Jarvis stopped with an error. Read the messages above.
goto :fail

:fail
echo.
pause
endlocal
exit /b 1
