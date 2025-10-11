@echo off
REM === QuantML Git Auto Commit Script ===
REM Usage: double-click or run from VS terminal after edits

REM --- Configuration (edit to match your repo) ---
set REPO_PATH=D:\QuantML\VS_QuantML_PROD_US
set COMMIT_MSG=Auto-update QuantML files

REM --- Move into repo ---
cd /d "%REPO_PATH%"

echo.
echo 🔄 Pulling latest changes from origin/main...
git pull origin main

echo.
echo 📦 Staging all modified files...
git add .

echo.
echo 📝 Committing with message: "%COMMIT_MSG%"
git commit -m "%COMMIT_MSG%"

echo.
echo 🚀 Pushing to remote repository...
git push origin main

echo.
echo ✅ Done! Changes synced with GitHub.
pause
