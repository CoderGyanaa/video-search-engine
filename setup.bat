@echo off
echo ============================================
echo  VideoSearch AI - Windows Setup Script
echo ============================================
echo.

REM Check Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install from https://python.org ^(3.9+^)
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo [4/5] Installing dependencies (this may take 5-10 mins first time)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency install failed. Check your internet connection.
    pause
    exit /b 1
)

echo [5/5] Creating required directories...
mkdir index 2>nul
mkdir thumbnails 2>nul
mkdir logs 2>nul
mkdir outputs 2>nul

echo.
echo ============================================
echo  Setup Complete!
echo ============================================
echo.
echo To run the Streamlit UI:
echo   venv\Scripts\activate
echo   streamlit run app.py
echo.
echo To use the CLI:
echo   venv\Scripts\activate
echo   python cli.py index C:\path\to\video.mp4
echo   python cli.py search "person carrying a bag"
echo.
pause
