@echo off

:: ASCII Art Generator Launcher (Windows)
:: =======================================
:: 
:: This script activates the virtual environment and launches the Streamlit app
:: 
:: Usage:
::   run.bat
::   or
::   double-click run.bat

title ASCII.MICR - Local ASCII Art Generator

echo ╔════════════════════════════════════════════════════════╗
echo ║          ASCII.MICR - Local ASCII Art Generator        ║
echo ╚════════════════════════════════════════════════════════╝
echo.

:: Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    
    echo Installing dependencies...
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install torch pillow numpy tqdm streamlit watchdog
) else (
    echo [OK] Virtual environment found
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if gradscii-art exists
if not exist "gradscii-art" (
    echo Cloning gradscii-art repository...
    git clone https://github.com/stong/gradscii-art.git
)

echo.
echo Starting ASCII Art Generator...
echo The application will open in your browser automatically
echo.

:: Launch Streamlit
streamlit run app.py --server.port=8501 --server.address=localhost

:: Deactivate when done
call venv\Scripts\deactivate.bat
