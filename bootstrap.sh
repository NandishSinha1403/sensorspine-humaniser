#!/bin/bash

# Humaniser Project Bootstrap Script
# This script automates the setup of both backend and frontend environments.

set -e # Exit on error

echo "🚀 Starting Humaniser Bootstrap Sequence..."

# 1. Environment Check
echo "🔍 Checking system requirements..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed."
    exit 1
fi

# 2. Backend Setup
echo "🐍 Setting up Backend (FastAPI)..."
cd humaniser/backend

# Install dependencies
echo "📦 Installing Python packages..."
python3 -m pip install -r requirements.txt

# Download NLP Models
echo "🧠 Downloading spaCy 'en_core_web_sm' model..."
python3 -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

echo "📚 Downloading NLTK data resources..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords')"

cd ../..

# 3. Frontend Setup
echo "⚛️ Setting up Frontend (Next.js)..."
cd humaniser/frontend

echo "📦 Installing npm packages..."
npm install

# Ensure PostCSS is configured (Fix for common Tailwind issues)
if [ ! -f postcss.config.js ]; then
    echo "🛠️ Creating missing postcss.config.js..."
    echo "module.exports = { plugins: { tailwindcss: {}, autoprefixer: {}, } }" > postcss.config.js
fi

cd ../..

echo ""
echo "✅ Bootstrap Complete!"
echo "------------------------------------------------"
echo "To run the application:"
echo ""
echo "1. Start Backend:"
echo "   cd humaniser/backend && python3 -m uvicorn app.main:app --reload --port 8000"
echo ""
echo "2. Start Frontend:"
echo "   cd humaniser/frontend && npm run dev"
echo "------------------------------------------------"
