@REM ./run_app.bat
@echo off
echo Starting Disease Prediction System and Database Monitor...
start "Disease Prediction App" cmd /k ".venv\Scripts\python.exe" app.py
start "Database Monitor" cmd /k ".venv\Scripts\python.exe" view_db.py
echo Both processes have been started in separate windows.
pause
