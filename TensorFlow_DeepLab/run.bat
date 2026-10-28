@echo off
cd /d %~dp0
where python >nul 2>nul && (set PY=python) || (set PY=py)
if not exist venv (%PY% -m venv venv)
call venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
pause
