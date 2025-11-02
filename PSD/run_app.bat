@echo off
echo Setting up Streamlit Voice Recognition App...

echo.
echo [1/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/4] Checking model file...
if not exist "voice_classifier_pipeline.pkl" (
    echo WARNING: Model file not found!
    echo Please run the training notebook first to generate the model.
    echo.
    pause
    exit /b 1
)

echo.
echo [3/4] Testing Streamlit installation...
streamlit --version

echo.
echo [4/4] Starting Streamlit app...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
pause
streamlit run streamlit_app.py