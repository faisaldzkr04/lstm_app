@echo off
title Aplikasi Analisis Sentimen LSTM
echo ========================================================
echo Membuka Aplikasi Streamlit Analisis Sentimen (LSTM)...
echo ========================================================
echo.
python -m streamlit run app.py --server.port 8501 --server.address localhost
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Terjadi kendala saat menjalankan aplikasi.
    echo Pastikan library sudah terinstall dengan menjalankan install.bat terlebih dahulu.
)
pause
