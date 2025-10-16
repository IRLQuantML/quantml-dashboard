REM 1) Make sure .gitignore ignores your large artifacts (*.csv, *.xlsx, *.pdf, *.png, *.joblib, *.pkl, logs, etc.)

REM 2) Reset your local branch to the remote tip, but keep your changes in the working tree
git fetch origin
git reset --soft origin/main

REM 3) Clear the index (nothing staged)
git restore --staged .

REM 4) Stage ONLY what you want to publish
git add hReview_Summary.py .gitignore README.md

REM 5) Sanity check before committing
git status
git diff --name-only --cached

REM 6) Commit and push
git commit -m "Publish dashboard-only changes"
git push origin main
pause
