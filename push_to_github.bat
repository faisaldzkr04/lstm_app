@echo off
echo ===================================================
echo  Pushing Proyek LSTM App ke GitHub...
echo ===================================================

git init
git remote remove origin 2>nul
git remote add origin https://github.com/faisaldzkr04/lstm_app.git
git branch -M main
git add .
git commit -m "Initial commit: Aplikasi Analisis Sentimen LSTM Timnas Indonesia"
git push -u origin main --force

echo ===================================================
echo  Selesai! Silakan periksa repository GitHub Anda:
echo  https://github.com/faisaldzkr04/lstm_app
echo ===================================================
pause
