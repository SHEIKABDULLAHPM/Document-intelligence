📄 AI Document Intelligence Assistant
https://img.shields.io/badge/python-3.11-blue.svg
https://img.shields.io/badge/streamlit-1.28.0-red.svg
https://img.shields.io/badge/docker-ready-blue.svg
https://img.shields.io/badge/license-MIT-green.svg

A powerful AI-powered document intelligence assistant that extracts, analyzes, and answers questions about your documents using state-of-the-art NLP and OCR technologies.

✨ Features
📄 Document Processing - Supports PDF, PNG, JPG, JPEG files

🔍 OCR Extraction - Extracts text from scanned documents and images

🤖 AI Summarization - Generates concise summaries using transformer models

🎯 Smart Information Extraction - Extracts entities, dates, names, locations, and keywords

💬 Interactive Q&A - Ask questions about your document and get intelligent answers

📊 Document Analytics - Get statistics and insights about your documents

🐳 Docker Ready - Easy deployment with Docker

🏗️ Architecture
text
┌─────────────────────────────────────────────────────────────┐
│                    📱 STREAMLIT UI LAYER                    │
├─────────────┬───────────────────────┬─────────────────────┤
│  📤 Document│  🔍 Information       │  💬 Q&A System      │
│    Upload   │    Extraction         │  (Semantic Search   │
│             │   (spaCy NER)         │   + BERT QA)        │
├─────────────┴───────────────────────┴─────────────────────┤
│              📸 OCR LAYER (Tesseract)                      │
├─────────────────────────────────────────────────────────────┤
│              🧠 NLP LAYER (Transformers)                   │
│   - BART Summarization                                    │
│   - Sentence Transformers                                 │
│   - DistilBERT QA                                        │
└─────────────────────────────────────────────────────────────┘
🚀 Quick Start
Prerequisites
Python 3.11+ or Docker

4GB+ RAM (8GB recommended)

Git

Option 1: Run with Docker (Recommended)
bash
# Clone the repository
git clone https://github.com/yourusername/document-intelligence.git
cd document-intelligence

# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up --build -d

# Access the application
# Open http://localhost:8501
Option 2: Run Locally
bash
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
📁 Project Structure
text
document_intelligence/
├── app/                         # Application source code
│   ├── main.py                 # Streamlit UI entry point
│   ├── ocr_engine.py           # OCR processing
│   ├── text_processor.py       # Text preprocessing
│   ├── summarizer.py           # AI summarization
│   ├── entity_extractor.py     # NER and information extraction
│   └── qa_system.py            # Q&A system
├── docker/                      # Docker configuration
│   ├── Dockerfile              # Docker image definition
│   └── docker-compose.yml      # Docker Compose configuration
├── requirements.txt             # Python dependencies
├── .dockerignore               # Docker ignore file
├── .gitignore                  # Git ignore file
├── README.md                   # This file
└── LICENSE                     # MIT License
🛠️ Technology Stack
Layer	Technology	Purpose
Frontend	Streamlit 1.28.0	Web interface
OCR	Tesseract 5.3.0	Text extraction from images
Image Processing	OpenCV 4.8.0 + Pillow 10.0.0	Image preprocessing
NLP	spaCy 3.5.0	Named Entity Recognition
Summarization	BART (Transformers 4.30.2)	Abstractive summarization
Q&A	DistilBERT + Sentence Transformers	Question answering
ML Backend	PyTorch 2.0.1	Deep learning
Data Processing	Pandas + NumPy	Data manipulation
Deployment	Docker + Docker Compose	Containerization
📊 How It Works
Data Flow
Upload → User uploads PDF or image document

OCR → Tesseract extracts text from the document

Process → Text is cleaned and preprocessed

Analyze → NLP pipeline extracts entities, keywords, and relationships

Summarize → BART model generates a concise summary

Index → Document chunks are indexed for Q&A

Q&A → Semantic search + DistilBERT answers user questions

Display → All insights presented in the UI

Key Components
OCR Engine: Extracts text from scanned documents and images

Entity Extractor: Identifies people, organizations, dates, locations

Summarizer: Generates human-like summaries using BART

Q&A System: Answers questions using semantic search and BERT

🐳 Docker Deployment
Build and Run
bash
# Build the Docker image
docker build -t document-intelligence:latest -f docker/Dockerfile .

# Run the container
docker run -d -p 8501:8501 --name doc-intel document-intelligence:latest

# View logs
docker logs -f doc-intel
Docker Compose
bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Stop all services
docker-compose -f docker/docker-compose.yml down

# View logs
docker-compose -f docker/docker-compose.yml logs -f
📝 Environment Variables
Create a .env file in the project root:

env
# Application Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Tesseract Configuration
TESSERACT_PATH=/usr/bin/tesseract

# Model Configuration
MODEL_CACHE_DIR=/app/cache
HF_HOME=/app/cache/huggingface

# Logging
LOG_LEVEL=INFO
🧪 Testing
bash
# Run tests
python -m pytest tests/

# Test Docker container
python test_docker.py
🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Streamlit - UI Framework

Hugging Face - Transformer Models

spaCy - NLP Library

Tesseract - OCR Engine

📞 Support
Issues: GitHub Issues

Email: your.email@example.com

⭐ Show Your Support
If you found this project helpful, please give it a ⭐ on GitHub!

Made with ❤️ by [Your Name]

