@echo off
setlocal
title Local OCR - Office Edition
echo ===================================================
echo   🚀 Local OCR Office - Startup Script
echo ===================================================

:: Set Environment Variables
set PYTHONIOENCODING=utf-8
set FLAGS_use_mkldnn=0

echo [INFO] Environment variables set.
echo [INFO] Opening GUI...
start gui_demo.html
echo [INFO] Starting API Server...
echo.

:: Run the API
python ocr_api.py

pause
