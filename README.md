# 🛡️ TamilGuard AI — Neural Abusive Language Detection & Moderation

**A production-ready, real-time abusive language detection system for Tamil, Tanglish (Tamil written in English), and English text.** Built with a BiLSTM-CRF sequence tagger, CharCNN + FastText dual feature extraction, and a curated lexicon safety net.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Neural Sequence Tagger** | BiLSTM-CRF with BIO tagging — identifies exact abusive spans, not just sentence labels |
| 🔤 **Dual Embeddings** | CharCNN (spelling variants) + FastText (subword semantics) — robust to Tanglish obfuscation |
| 📖 **Lexicon Safety Net** | 275-key curated lexicon with suffix stripping (`-kku`, `-la`, `-nga`) for 100% recall on known slurs |
| 🎭 **Smart Redaction** | 4 modes: Subtle Mask (`th***ya`), Category Tag (`[REDACTED: SLUR]`), Full Block (`████`), Polite Rephrase |
| ⚡ **Real-time** | ~18ms per sentence on CPU |
| 🌐 **Full-stack** | React frontend + Flask API + Chrome extension |
| 🔧 **3 Sensitivity Levels** | Standard, Strict (leetspeak + fuzzy), Maximum (zero tolerance) |

---

## 📐 Architecture

```
Input Text
    │
    ▼
┌─────────────────────┐
│  Unicode Normalizer  │  NFC, HTML decode, repeat collapse
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Dual Feature Extractor  │
│  FastText (200D)    │  Subword semantic vectors
│  CharCNN (150D)     │  Spelling variant capture
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Bidirectional LSTM  │  256-hidden contextual encoder
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  CRF Decoder         │  Viterbi BIO tag decoding
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Lexicon Override    │  OR-merge with safety net
└─────────┬───────────┘
          ▼
   InferenceResult
   (safe/unsafe, spans, categories)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+ (for frontend)
- Trained model checkpoint at `models/checkpoints/sequence_tagger/best.pt`
- FastText embeddings at `embeddings/`

### 1. Backend Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run build    # Build for production
cd ..
```

### 3. Run the Server

```bash
# Development (Flask dev server)
python serve.py --web

# Production (Gunicorn)
gunicorn serve:app --bind 0.0.0.0:5000 --workers 1 --timeout 120
```

The server will:
- Serve the React frontend at `http://localhost:5000/`
- Expose the API at `http://localhost:5000/predict`
- Show the built-in HTML demo if frontend isn't built

### 4. CLI Usage

```bash
# Single text
python serve.py "நீ ஒரு thevdiya da"

# With censoring
python serve.py --censor --censor-mode partial "நீ ஒரு thevdiya da"

# Batch from file
python serve.py --file input.txt

# Interactive mode
python serve.py
```

---

## 📡 API Reference

### `POST /predict`
Analyze text for abusive content.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "நீ ஒரு thevdiya da", "sensitivity": "standard"}'
```

**Response:**
```json
{
  "safe": false,
  "flagged_words": ["thevdiya"],
  "categories": ["sexual"],
  "flagged_spans": [
    {
      "token": "thevdiya",
      "source": "both",
      "canon": "thevidiya",
      "category": "sexual",
      "severity": "high",
      "match_type": "exact"
    }
  ],
  "_ms": 18.2
}
```

Add `?full=true` for token-level debug output.

### `POST /censor`
Auto-censor abusive content.

```bash
curl -X POST http://localhost:5000/censor \
  -H "Content-Type: application/json" \
  -d '{"text": "நீ ஒரு thevdiya da", "mode": "partial"}'
```

**Modes:** `partial` | `tag` | `block` | `polite`

### `POST /predict_batch`
Batch analysis of multiple texts.

```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["text1", "text2", "text3"]}'
```

### `GET /health`
Health check endpoint.

---

## 🌍 Deployment

### Option 1: Docker (Recommended)

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t tamilguard-ai .
docker run -p 5000:5000 tamilguard-ai
```

### Option 2: Render.com

1. Push your code to GitHub (include model checkpoint)
2. Connect your repo on [render.com](https://render.com)
3. Use the included `render.yaml` for auto-configuration
4. Or click: **Deploy to Render** → select Docker environment

> **Note:** Use the Starter plan ($7/mo) minimum — PyTorch needs ~512MB RAM.

### Option 3: Railway

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects the Dockerfile
4. Set environment variable `PORT=5000`

### Option 4: Hugging Face Spaces (Free GPU!)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Upload your project files including the Dockerfile
4. HF provides free CPU (and even free GPU for eligible users)

### Option 5: VPS (DigitalOcean / Linode)

```bash
# On your VPS (Ubuntu)
sudo apt update && sudo apt install docker.io docker-compose -y
git clone <your-repo> && cd <your-repo>
docker-compose up -d --build
```

Recommended: DigitalOcean Droplet ($6/mo, 1GB RAM) or Linode ($5/mo).

---

## 📊 Benchmarks

| Metric | Score |
|---|---|
| Gold Test Abuse F1 (token-level) | **96.79%** |
| Precision | **100.0%** (0 false positives on benign text) |
| Sentence-level Unsafe F1 | **96.59%** |
| Average Latency (CPU) | **~18ms** |
| Model Parameters | **7.6M** |

---

## 🗂️ Project Structure

```
Abusive Detection/
├── serve.py                 # Flask API server + CLI
├── inference.py             # End-to-end inference pipeline
├── normalize.py             # Unicode & script normalization
├── redact.py                # Smart redaction / censoring engine
├── torchcrf.py              # CRF layer implementation
├── requirements.txt         # Python dependencies
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Local Docker deployment
├── render.yaml              # Render.com blueprint
├── Procfile                 # Gunicorn startup for PaaS
│
├── models/
│   ├── sequence_tagger.py   # BiLSTM-CRF model definition
│   ├── sequence_vocab.json  # Vocabulary mappings
│   └── checkpoints/         # Trained model weights
│
├── lexicon/
│   └── abusive_lexicon.json # Curated abuse lexicon (275 keys)
│
├── frontend/                # React + Vite + Tailwind dashboard
│   ├── src/
│   │   ├── components/      # LiveAnalyzer, BatchTester, etc.
│   │   ├── services/api.js  # API client with offline fallback
│   │   └── data/            # Test presets
│   └── dist/                # Production build output
│
├── browser-extension/       # Chrome extension for real-time filtering
│
├── scripts/                 # Training, evaluation, data prep scripts
│   ├── train_sequence_tagger.py
│   ├── evaluate_gold_test.py
│   ├── build_lexicon.py
│   └── ...
│
└── Datasets/                # Training & evaluation data
```

---

## 🔧 Browser Extension

The Chrome extension provides real-time content filtering on any webpage.

### Installation
1. Open Chrome → `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" → select the `browser-extension/` folder

---

## 📄 License

This project is for educational and research purposes.

---

**Built with ❤️ for Tamil internet safety**
