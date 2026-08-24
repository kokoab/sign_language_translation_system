@echo off
setlocal enabledelayedexpansion

REM ── Always run from the folder this .bat lives in ───────
cd /d "%~dp0"

echo ============================================
echo  SLT - Data Collection Setup (Windows)
echo  One-click: Python + dependencies + run
echo ============================================
echo.
echo  Working directory: %cd%
echo.

set PYTHON_VERSION=3.10.11
set INSTALLER_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe
set INSTALLER_FILE=_python_installer.exe
set DEFAULT_INSTALL=%LOCALAPPDATA%\Programs\Python\Python310
set PYTHON_CMD=

REM ── Step 1: Find a usable Python 3.10+ ─────────────────

REM Check PATH first
where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do (
        echo %%v | findstr /R "^3\.1[0-9]" >nul
        if !errorlevel! equ 0 (
            set PYTHON_CMD=python
            echo [OK] Found Python %%v in PATH.
        )
    )
)

REM Check default per-user install location
if "!PYTHON_CMD!"=="" (
    if exist "%DEFAULT_INSTALL%\python.exe" (
        set "PYTHON_CMD=%DEFAULT_INSTALL%\python.exe"
        echo [OK] Found Python at %DEFAULT_INSTALL%
    )
)

REM ── Step 2: Install Python if not found ─────────────────

if "!PYTHON_CMD!"=="" (
    echo.
    echo [INFO] Python 3.10+ not found on this system.
    echo        Downloading Python %PYTHON_VERSION% installer...
    echo.

    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%INSTALLER_FILE%' }"
    if not exist "%INSTALLER_FILE%" (
        echo [ERROR] Download failed. Check your internet connection.
        echo         You can also install Python manually from:
        echo         https://www.python.org/downloads/
        goto :fail
    )

    echo [INFO] Installing Python %PYTHON_VERSION% (per-user, no admin needed)...
    echo        This adds Python to your PATH automatically.
    echo.
    %INSTALLER_FILE% /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0
    if errorlevel 1 (
        echo [ERROR] Python installation failed.
        del /f "%INSTALLER_FILE%" 2>nul
        goto :fail
    )

    del /f "%INSTALLER_FILE%" 2>nul
    echo [OK] Python %PYTHON_VERSION% installed.

    REM Point to the known install location since PATH won't refresh in this session
    if exist "%DEFAULT_INSTALL%\python.exe" (
        set "PYTHON_CMD=%DEFAULT_INSTALL%\python.exe"
    ) else (
        echo [ERROR] Could not locate Python after installation.
        echo         Close this window, reopen, and run setup_collect.bat again.
        goto :fail
    )
)

echo.
echo Using: !PYTHON_CMD!
for /f "tokens=*" %%v in ('"!PYTHON_CMD!" --version 2^>^&1') do echo        %%v
echo.

REM ── Step 3: Create virtual environment ──────────────────

if not exist "venv_collect" (
    echo Creating virtual environment...
    "!PYTHON_CMD!" -m venv venv_collect
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        goto :fail
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists, reusing it.
)

REM ── Step 4: Install dependencies ────────────────────────

echo.
echo Installing dependencies...
call venv_collect\Scripts\activate.bat

python -m pip install --upgrade pip --quiet
pip install -r requirements_collect.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    goto :fail
)
echo [OK] All dependencies installed.

REM ── Step 5: Run ─────────────────────────────────────────

echo.
echo ============================================
echo  Setup complete! Launching data collection
echo ============================================
echo.
echo  Controls:
echo    Type label + Enter, then video count + Enter
echo    SPACE = start/stop recording
echo    O     = save clip
echo    U     = undo last clip
echo    Q     = quit
echo.

python src\collect_data.py

echo.
echo Done. Videos saved to data\raw_videos\
echo.
pause
exit /b 0

:fail
echo.
echo ============================================
echo  Setup failed. See the error above.
echo ============================================
echo.
pause
exit /b 1
