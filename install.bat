@echo off

echo Installing required packages from requirements.txt...
pip install -r requirements.txt

echo Running template.py to generate project structure...
python template.py

echo Setup complete!
pause
