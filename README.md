# 📄 AI Document Intelligence Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A powerful **AI-powered Document Intelligence Assistant** that extracts, analyzes, summarizes, and answers questions from documents using advanced **OCR, NLP, and Transformer-based AI models**.

---

## ✨ Features

- 📄 **Document Processing**
  - Supports PDF, PNG, JPG, and JPEG files.

- 🔍 **OCR Text Extraction**
  - Extracts text from scanned documents and images using Tesseract OCR.

- 🤖 **AI Summarization**
  - Generates concise document summaries using Transformer models.

- 🎯 **Smart Information Extraction**
  - Identifies entities such as names, organizations, locations, dates, and keywords.

- 💬 **Interactive Question Answering**
  - Ask questions about uploaded documents and receive intelligent answers.

- 📊 **Document Analytics**
  - Provides document statistics and insights.

- 🐳 **Docker Ready**
  - Easy deployment using Docker and Docker Compose.

---

# 🏗️ Architecture

```text
Input Document
      │
      ▼
┌─────────────────────────────────────────────┐
│           PREPROCESSING                     │
│  ┌──────────────────────────────────────┐  │
│  │  Image/PDF → Text Extraction (OCR)   │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │  Text Cleaning & Normalization       │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│           ANALYSIS LAYER                    │
│  ┌──────────────┐  ┌────────────────────┐  │
│  │  NER         │  │  Keyword Extraction │  │
│  │  (spaCy)     │  │  (TF-IDF)          │  │
│  └──────────────┘  └────────────────────┘  │
│  ┌──────────────┐  ┌────────────────────┐  │
│  │  Relation    │  │  Fact Extraction   │  │
│  │  Extraction  │  │  (Pattern-based)   │  │
│  └──────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│           SUMMARIZATION                      │
│  ┌──────────────────────────────────────┐  │
│  │  BART Model (Abstractive)            │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│           INDEXING & Q&A                    │
│  ┌──────────────────────────────────────┐  │
│  │  Semantic Indexing (Embeddings)      │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │  QA Model (DistilBERT)               │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
      │
      ▼
   Output Results
```

---

# 🚀 Quick Start

## Prerequisites

- Python 3.11+
- Docker (Optional)
- Git
- 4GB RAM (8GB Recommended)

---

## Option 1 — Run with Docker

```bash
# Clone repository
git clone https://github.com/YOUR_GITHUB_USERNAME/AI-Document-Intelligence-Assistant.git

cd AI-Document-Intelligence-Assistant

# Build and run
docker-compose -f docker/docker-compose.yml up --build -d

# Open
http://localhost:8501
```

---

## Option 2 — Run Locally

```bash
# Clone repository
git clone https://github.com/YOUR_GITHUB_USERNAME/AI-Document-Intelligence-Assistant.git

cd AI-Document-Intelligence-Assistant

# Create virtual environment
python -m venv venv

# Activate environment

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run application
streamlit run app/main.py

# Open browser
http://localhost:8501
```

---

# 📁 Project Structure

```text
AI-Document-Intelligence-Assistant/
│
├── app/
│   ├── main.py
│   ├── ocr_engine.py
│   ├── text_processor.py
│   ├── summarizer.py
│   ├── entity_extractor.py
│   └── qa_system.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt
├── .dockerignore
├── .gitignore
├── README.md
└── LICENSE
```

---

# 🛠️ Technology Stack

| Layer | Technology |
|---------|------------|
| Frontend | Streamlit |
| OCR | Tesseract OCR |
| Image Processing | OpenCV, Pillow |
| NLP | spaCy |
| Summarization | BART (Transformers) |
| Question Answering | DistilBERT + Sentence Transformers |
| Machine Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Deployment | Docker |

---

# 📊 Workflow

```text
Upload Document
        │
        ▼
OCR Extraction
        │
        ▼
Text Preprocessing
        │
        ▼
Entity Recognition
        │
        ▼
AI Summarization
        │
        ▼
Semantic Indexing
        │
        ▼
Question Answering
        │
        ▼
Interactive Dashboard
```

---

# 🧠 Core Components

### 📸 OCR Engine

Extracts text from scanned documents and images.

### 🎯 Entity Extractor

Recognizes:

- Person Names
- Organizations
- Dates
- Locations
- Keywords

### 🤖 Summarizer

Uses Facebook BART model to generate high-quality summaries.

### 💬 Question Answering

Combines Semantic Search with DistilBERT for contextual answers.

---

# 🐳 Docker Deployment

## Build Docker Image

```bash
docker build -t document-intelligence -f docker/Dockerfile .
```

## Run Container

```bash
docker run -d -p 8501:8501 --name document-ai document-intelligence
```

## View Logs

```bash
docker logs -f document-ai
```

---

## Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up -d
```

Stop services

```bash
docker-compose -f docker/docker-compose.yml down
```

View logs

```bash
docker-compose -f docker/docker-compose.yml logs -f
```

---

# ⚙️ Environment Variables

Create a `.env` file.

```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

TESSERACT_PATH=/usr/bin/tesseract

MODEL_CACHE_DIR=/app/cache
HF_HOME=/app/cache/huggingface

LOG_LEVEL=INFO
```

---

# 🧪 Testing

```bash
python -m pytest tests/
```

Docker test

```bash
python test_docker.py
```

---

# 🤝 Contributing

1. Fork the repository

2. Create a new branch

```bash
git checkout -b feature/new-feature
```

3. Commit changes

```bash
git commit -m "Add new feature"
```

4. Push changes

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgements

- Streamlit
- Hugging Face Transformers
- spaCy
- Tesseract OCR
- OpenCV
- PyTorch

---

# 👨‍💻 Developer

**PEER SHEIK ABDULLAH MOHD NOORDEEN P M**

**B.Tech Computer Science and Business Systems (CSBS)**

- 💼 Aspiring AI/ML Engineer
- 🤖 Generative AI & NLP Enthusiast
- 🌐 Full Stack Developer
- 📊 Machine Learning Developer

---

# 📫 Connect with Me

**GitHub**

```
https://github.com/YOUR_GITHUB_USERNAME
```

**LinkedIn**

```
https://linkedin.com/in/YOUR_LINKEDIN_USERNAME
```

**Portfolio**

```
https://YOUR_PORTFOLIO_URL
```

---

# ⭐ Support

If you found this project useful, please consider giving it a **⭐ on GitHub**.

It helps support the project and encourages future development.

---

<div align="center">

### Made with ❤️ by **PEER SHEIK ABDULLAH MOHD NOORDEEN P M**

</div>
