@echo off
REM Quick launcher for Streamlit UI
call venv\Scripts\activate.bat
echo Starting VideoSearch AI UI at http://localhost:8501
streamlit run app.py
