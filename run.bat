@echo off

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
pip install -r requirements.txt

REM Create necessary directories
mkdir model 2>nul
mkdir static\uploads 2>nul
mkdir dataset\train\early_blight 2>nul
mkdir dataset\train\late_blight 2>nul
mkdir dataset\train\healthy 2>nul
mkdir dataset\validation\early_blight 2>nul
mkdir dataset\validation\late_blight 2>nul
mkdir dataset\validation\healthy 2>nul

REM Run the Flask app
python app.py

pause