@echo off
setlocal

REM === Path & env ===
cd /d D:\QuantML\VS_QuantML_PROD_US
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"

REM === Prefer config over any args inside the script ===
set "QML_PREFER_CONFIG=1"

REM === (Optional) choose a profile defined in config.py ===
REM set "TP_EXTEND_PROFILE=balanced"
REM set "TP_EXTEND_PROFILE=aggressive"
REM set "TP_EXTEND_PROFILE=conservative"

REM === Guard: avoid duplicate extender windows by title ===
for /f "tokens=*" %%p in ('tasklist /v /fi "windowtitle eq QuantML TP Extender (ATR)" ^| findstr /i /c:"QuantML TP Extender (ATR)"') do (
  echo Extender already running. Skipping new instance.
  goto :eof
)

REM === Start extender (NO CLI OVERRIDES) ===
start "QuantML TP Extender (ATR)" cmd /k python -u fATR_trading.py

endlocal
