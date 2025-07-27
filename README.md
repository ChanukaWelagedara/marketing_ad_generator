# 🚀 AI-Powered Marketing Ad Generator

A fine-tuned GPT-2 model for generating personalized Facebook ads with GPU acceleration support.

## 📋 Table of Contents

- [Dependencies & Installation](#dependencies--installation)
- [GPU Setup](#gpu-setup)
- [Complete Workflow](#complete-workflow)
- [Usage](#usage)
- [Project Structure](#project-structure)

## 🔧 Dependencies & Installation

### 1. System Requirements

- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **CUDA**: 11.8+ or 12.0+
- **Memory**: 8GB+ GPU memory recommended

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install PyTorch with CUDA support (choose your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install -r requirements.txt
```

### 3. Required Dependencies (`requirements.txt`)

```txt
# Core ML Libraries
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0

# NLP Processing
spacy>=3.6.0
sentence-transformers>=2.2.0
textblob>=0.17.1
langdetect>=1.0.9

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Web Interface
streamlit>=1.24.0

# Utilities
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0

# spaCy Model (install separately)
# python -m spacy download en_core_web_sm
```

### 4. Install spaCy Model

```bash
# Install English model for spaCy
python -m spacy download en_core_web_sm

# For GPU acceleration (optional)
pip install spacy[cuda118]  # or cuda120
python -m spacy download en_core_web_trf
```

## ⚡ GPU Setup

### 1. Enable GPU Acceleration (Run Once)

```bash
python gpu_accelerator.py
```

## 🔄 Complete Workflow

Follow these steps in order to train the model

### Step 1: Data Preparation

```bash
# Generate enhanced dataset with embeddings
python scripts\generate_dataset.py

# Apply NLP preprocessing and feature extraction
python scripts\nlp_preprocessing.py

# Generate embeddings for clustering
python scripts\generate_embeddings.py
```

### Step 2: Create Training Data

```bash
# Create prompts for fine-tuning
python scripts\create_prompts.py
```

### Step 3: Fine-tune Model

```bash
# Train the GPT-2 model on your data
python scripts\train_finetune.py
```

### Step 4: Test Model

```bash
# Test the fine-tuned model
python test_model.py

# Or test with custom generation
python generate_ad_text.py



### Model Paths

- **Fine-tuned model**: `./models/fine_tuned_gpt2_clustered/`
- **Checkpoints**: `./models/checkpoint-*/`
- **Data**: `./data/`

## 📁 Project Structure

```

marketing_ad_generator/
├── 📄 README.md # This file
├── 📄 requirements.txt # Dependencies
├── 📄 .gitignore # Git ignore rules
├── 🚀 gpu_accelerator.py # GPU setup script
├── 🧪 test_model.py # Model testing
├── 📝 generate_ad_text.py # Ad generation script
│
├── 📱 app/
│ └── app.py # Streamlit web interface
│
├── 📊 data/
│ ├── amazon_ads_dataset.csv # Raw dataset
│ ├── enhanced_amazon_ads.csv # Processed dataset
│ ├── amazon_ads_with_clusters.json # Clustered data
│ └── train_data_clustered_prompts.jsonl # Training data
│
├── 🤖 models/
│ ├── fine_tuned_gpt2_clustered/ # Final trained model
│ └── checkpoint-\*/ # Training checkpoints
│
├── 📜 scripts/
│ ├── generate_dataset.py # Data generation
│ ├── nlp_preprocessing.py # Text processing
│ ├── generate_embeddings.py # Embedding creation
│ ├── create_prompts.py # Training prompt creation
│ └── train_finetune.py # Model training
│
└── 📓 notebooks/ # Jupyter notebooks (optional)

```

```
