@echo off
setlocal enabledelayedexpansion

echo 🚀 Starting Humaniser Bootstrap Sequence...

:: 1. Environment Check
echo 🔍 Checking system requirements...

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: python is not installed.
    exit /b 1
)

where npm >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: npm is not installed.
    exit /b 1
)

:: 2. Backend Setup
echo 🐍 Setting up Backend (FastAPI)...
pushd humaniser\backend

echo 📦 Installing Python packages...
python -m pip install -r requirements.txt

echo 🧠 Downloading spaCy 'en_core_web_sm' model...
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

echo 📚 Downloading NLTK data resources...
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords')"

popd

:: 3. Frontend Setup
echo ⚛️ Setting up Frontend (Next.js)...
pushd humaniser\frontend

echo 📦 Installing npm packages...
npm install

:: Ensure PostCSS is configured
if not exist postcss.config.js (
    echo 🛠️ Creating missing postcss.config.js...
    echo module.exports = { plugins: { tailwindcss: {}, autoprefixer: {}, } } > postcss.config.js
)

popd

echo.
echo ✅ Bootstrap Complete!
echo ------------------------------------------------
echo To run the application:
echo.
echo 1. Start Backend:
echo    cd humaniser\backend ^&^& python -m uvicorn app.main:app --reload --port 8000
echo.
echo 2. Start Frontend:
echo    cd humaniser\frontend ^&^& npm run dev
echo ------------------------------------------------
pause
