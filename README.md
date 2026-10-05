# AuthentiX AI: Multimodal Deepfake & Phishing Detection System

[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Deep Learning](https://img.shields.io/badge/ML-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Database](https://img.shields.io/badge/Database-PostgreSQL_16-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)

**AuthentiX AI** is an evidence-based, multimodal digital forensics and web security system designed to detect synthetic media (deepfake images/videos) and phishing/malicious URL threats with transparent diagnostic metrics.

---

## 🏗️ System Architecture

AuthentiX AI uses a dual-pipeline fusion architecture:

```
                  ┌──────────────────────────────────────────────┐
                  │                 Unified Input                │
                  │        (Image / Video / URL Submission)      │
                  └──────────────────────┬───────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
         ┌─────────▼───────────┐                     ┌─────────▼───────────┐
         │     Media Track     │                     │      URL Track      │
         │ (Image / Video File)│                     │     (Target URL)    │
         └─────────┬───────────┘                     └─────────┬───────────┘
                   │                                           │
         ┌─────────▼───────────┐                     ┌─────────▼───────────┐
         │ Frame Ext. / Preproc│                     │ Feature Extractor   │
         │ (FFmpeg / OpenCV)   │                     │ (Lexical, Domain,   │
         └─────────┬───────────┘                     │  Security Signals)  │
                   │                                 └─────────┬───────────┘
         ┌─────────▼───────────┐                               │
         │  Fusion Classifier  │                     ┌─────────▼───────────┐
         │  EfficientNet-B4 +  │                     │   URL Classifier    │
         │  ViT/CLIP Features  │                     │  (XGBoost Engine)   │
         └─────────┬───────────┘                     └─────────┬───────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                  ┌──────────────────────▼───────────────────────┐
                  │         Multimodal Fusion & Analytics        │
                  │   Aggregated Authenticity Score & Report     │
                  └──────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | High-performance async support and deep learning ecosystem compatibility |
| **Web API** | FastAPI + Uvicorn | Asynchronous processing, native Pydantic validation, automatic OpenAPI docs |
| **Database** | PostgreSQL 16 | ACID-compliant storage for audit logs, prediction reports, and metadata |
| **Image Forensics** | EfficientNet-B4 + ViT/CLIP | State-of-the-art artifact analysis with feature-level late fusion |
| **URL Forensics** | XGBoost Engine | High accuracy on tabular lexical, domain age, SSL, and structural signals |
| **Video Extraction** | FFmpeg / OpenCV | High-speed multi-frame sampling for temporal consistency evaluation |

---

## 📁 Repository Structure

```
Authentix-AI/
├── configs/                # Configuration YAMLs (logging, model parameters)
├── scripts/                # Data cleaning, preprocessing & training pipelines
│   ├── clean_image_datasets.py
│   ├── clean_url_datasets.py
│   ├── run_preprocessing.py
│   └── train_url_classifier.py
├── src/                    # Core application source code
│   └── authentix/
│       ├── api/            # FastAPI routes & endpoints
│       ├── core/           # Configs, logging & Pydantic schemas
│       ├── db/             # SQLAlchemy ORM models & session setup
│       ├── models/         # ML classifiers (EfficientNet, XGBoost)
│       └── pipelines/      # Data cleaning & preprocessing logic
├── tests/                  # Unit and integration test suites
│   ├── unit/
│   └── integration/
├── .env.example            # Template for environment configuration
├── .gitignore              # Git ignore rules for data, weights & secrets
├── before review2.md       # Academic Review-2 defense document & roadmap
├── pyproject.toml          # Build system configuration
└── requirements.txt        # Python dependency declarations
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Tanmay-Chavan/Authentix-AI.git
cd Authentix-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and set local parameters:

```bash
cp .env.example .env
```

### 3. Run Tests

```bash
pytest tests/unit/
```

### 4. Data Cleaning & Preprocessing (Offline Stage M1.5)

Run automated dataset audit and sanitization:

```bash
python scripts/run_preprocessing.py
```

---

## 📋 Project Roadmap

- [x] **Milestone 1**: Project Initialization, Baseline Architecture & Core Schemas
- [x] **Milestone 1.5**: Data Pipeline Sanitization & Cleaning Engine (Corrupt File Purging, Outlier Removal, Class Balancing)
- [ ] **Milestone 2**: Feature Extraction & Fusion Layer Training (EfficientNet-B4 + ViT/CLIP + XGBoost)
- [ ] **Milestone 3**: System Integration & Async Batch Processing API
- [ ] **Milestone 4**: Final Evaluation, Metrics Dashboard & Documentation

---

## 📜 License & Citation

Developed for Academic Research & Digital Forensics Evaluation.  
**DMCE Senior Project — AuthentiX AI (2026)**
