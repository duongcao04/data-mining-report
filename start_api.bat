@echo off
title Customer Churn Prediction API
echo ========================================
echo  Khoi dong Customer Churn Prediction API
echo ========================================
echo.

REM Di chuyen den thu muc chua script (neu double-click)
cd /d "%~dp0"

REM Kich hoat virtual environment
IF EXIST "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) ELSE (
    echo [ERROR] Khong tim thay venv\Scripts\activate.bat
    echo Vui long tao venv va cai dat requirements truoc.
    echo Lenh goi y:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Dang khoi dong API tai http://127.0.0.1:8000
echo (Nhan Ctrl+C trong cua so nay de dung server)
echo.

uvicorn demo.app:app --reload

pause


