@echo off
REM ============================================================
REM  Ember / Neural - Local Server Launcher (Windows)
REM ============================================================

REM Default host and port (can be overridden by args or env)
set HOST=127.0.0.1
set PORT=8000

REM Allow overrides via command-line args
IF NOT "%~1"=="" set HOST=%~1
IF NOT "%~2"=="" set PORT=%~2

echo.
echo ===========================================
echo  Starting Ember Local Server
echo  Host: %HOST%
echo  Port: %PORT%
echo ===========================================
echo.

REM Activate virtual environment if present
IF EXIST venv\Scripts\activate (
    call venv\Scripts\activate
)

REM Run the FastAPI app with uvicorn
python -m uvicorn src.app:app --host %HOST% --port %PORT% --reload

REM Pause so console doesn’t close immediately if launched by double-click
echo.
pause