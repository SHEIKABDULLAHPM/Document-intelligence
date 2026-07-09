# 📄 AI Document Intelligence Assistant

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A powerful AI-powered document intelligence assistant that extracts, analyzes, and answers questions about your documents using state-of-the-art NLP and OCR technologies.

## ✨ Features

- 📄 **Document Processing**: Supports PDF, PNG, JPG, JPEG files
- 🔍 **OCR Extraction**: Extracts text from scanned documents and images
- 🤖 **AI Summarization**: Generates concise summaries using transformer models
- 🎯 **Smart Information Extraction**: Extracts entities, dates, names, locations, and keywords
- 💬 **Interactive Q&A**: Ask questions about your document and get intelligent answers
- 📊 **Document Analytics**: Get statistics and insights about your documents
- 🐳 **Docker Ready**: Easy deployment with Docker
- 🚀 **Production Ready**: Scalable and containerized architecture

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│ Streamlit UI Layer │
├─────────────┬───────────────────────┬─────────────────────┤
│ Document │ Information │ Q&A System │
│ Upload │ Extraction │ (Semantic Search │
│ │ (spaCy NER) │ + BERT QA) │
├─────────────┴───────────────────────┴─────────────────────┤
│ OCR Layer (Tesseract) │
├─────────────────────────────────────────────────────────────┤
│ NLP Layer (Transformers) │
│ - BART Summarization │
│ - Sentence Transformers │
│ - DistilBERT QA │
└─────────────────────────────────────────────────────────────┘


## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** or **Docker**
- **4GB+ RAM** (8GB recommended)
- **Git**

### Option 1: Run with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/document-intelligence.git
cd document-intelligence

# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Open http://localhost:8501

```
### Option 2: Run locally
```bash
# Clone the repository
git clone https://github.com/yourusername/document-intelligence.git
cd document-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the application
streamlit run app/main.py

# Access the application
# Open http://localhost:8501

```