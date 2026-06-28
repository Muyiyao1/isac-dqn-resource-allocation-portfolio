@echo off
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" run_experiment.py --no-ga --output-dir results/full_no_ga --model-dir models/full_no_ga
pause
