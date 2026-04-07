# Use a slim Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8000

# Install system dependencies needed for spacy and nltk
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY humaniser/backend/requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download SpaCy model
RUN python -m spacy download en_core_web_sm

# Pre-download NLTK resources
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('brown')"

# Copy the rest of the application code
COPY . .

# Generate the Brown frequency data during build
WORKDIR /app/humaniser/backend
RUN python scripts/generate_brown_data.py

# Final working directory for runtime
WORKDIR /app/humaniser/backend

# Expose the port
EXPOSE 8000

# Command to run the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
