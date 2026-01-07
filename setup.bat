@echo off
setlocal
title Local OCR - Setup
echo ===================================================
echo   🛠️ Local OCR Office - Setup Script
echo ===================================================

echo [INFO] Installing Python dependencies...
pip install -r requirements.txt

echo.
echo [INFO] Checking Ollama models...
echo [INFO] Pulling DeepSeek OCR (this might take a while if not present)...
ollama pull deepseek-ocr

echo.
echo ===================================================
echo   ✅ Setup Complete!
echo   You can now run 'run.bat' to start the application.
echo ===================================================
pause
