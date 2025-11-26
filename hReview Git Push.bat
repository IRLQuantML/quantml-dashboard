@echo off
chcp 65001 >nul
REM === QuantML Git Safe Auto Commit Script ===
REM Only pushes hReview_Summary.py, .gitignore, and this script.

REM --- Repo Config ---
set "REPO_PATH=D:\QuantML\quantml-dashboard"
set COMMIT_MSG=Auto-update QuantML dashboard only

cd /d "%REPO_PATH%"

echo.
echo 🔄 Pulling latest changes from origin/main...
git pull origin main

echo.
echo 📦 Staging only allowed files...
git add hReview_Summary.py .gitignore "hReview Git Push.bat"

echo.
echo 📝 Committing with message: "%COMMIT_MSG%"
git commit -m "%COMMIT_MSG%"

echo.
echo 🚀 Pushing to remote repository...
git push origin main

echo.
echo ✅ Done! Only dashboard file updated on GitHub.
pause
