# 📡 Fake News Detector Using Machine Learning and NLP

An end-to-end AI-powered application that detects fake news using an ensemble machine learning model, transformer-based NLP, and external verification APIs. Built with Streamlit for a clean, real-time interactive interface.

---

## 🚀 Features

- **Ensemble ML Prediction** — Combines Logistic Regression and XGBoost classifiers (probability voting) for robust fake/true classification
- **Sentiment Analysis** — Runs transformer-based sentiment analysis on articles predicted as true
- **Text Summarization** — Generates a suggested neutral rewrite for articles flagged as fake
- **Google Fact Check API** — Queries verified fact-check results for fake articles
- **Google Safe Browsing API** — Checks if a URL is flagged as malicious before extracting content
- **Multiple Input Modes** — Paste text, upload a CSV file, or enter a website URL
- **URL Content Extraction** — Automatically extracts article text from URLs using `trafilatura`
- **Batch Processing** — Analyze multiple articles at once via CSV upload with downloadable results

---

## 📁 Project Structure

```
fake-news-detector/
│
├── app.py                        # Main Streamlit application
│
├── models/
│   ├── pipe_lr.joblib            # Logistic Regression pipeline (TF-IDF + LR)
│   ├── pipe_xgb.joblib           # XGBoost pipeline (TF-IDF + XGBoost)
│   ├── ensemble.joblib           # Ensemble (ProbVotingClassifier) of both pipelines
│   ├── sentiment_model_name.pkl  # Name of the HuggingFace sentiment model to load
│   └── gen_model_name.pkl        # Name of the HuggingFace summarization model to load
│
└── requirements.txt              # Python dependencies
```

> ⚠️ Place all `.joblib` and `.pkl` files inside a `models/` folder in the same directory as `app.py`.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place model files

Create a `models/` folder and move all model files into it:

```bash
mkdir models
mv pipe_lr.joblib pipe_xgb.joblib ensemble.joblib \
   sentiment_model_name.pkl gen_model_name.pkl models/
```

---

## 🔑 API Keys Setup

This project uses two Google APIs. Replace the placeholder keys in `app.py` with your own:

```python
GOOGLE_SAFE_BROWSING_KEY = "YOUR_KEY_HERE"
GOOGLE_FACTCHECK_KEY     = "YOUR_KEY_HERE"
```

> ⚠️ Never commit real API keys to a public repository. Use environment variables or a `.env` file in production.

**How to get the keys:**
- **Google Safe Browsing API** → [https://developers.google.com/safe-browsing](https://developers.google.com/safe-browsing)
- **Google Fact Check Tools API** → [https://developers.google.com/fact-check/tools/api](https://developers.google.com/fact-check/tools/api)

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📋 Requirements

Create a `requirements.txt` with the following:

```
streamlit
scikit-learn
xgboost
joblib
transformers
torch
nltk
pandas
numpy
requests
trafilatura
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🧠 How It Works

### Model Pipeline

```
Input Text
    │
    ▼
TF-IDF Vectorizer
    │
    ├──▶ Logistic Regression  ──┐
    │                           ├──▶ Probability Voting ──▶ FAKE / TRUE
    └──▶ XGBoost Classifier   ──┘
```

### If Predicted TRUE
- Runs sentiment analysis using a HuggingFace transformer model
- Displays the sentiment label and confidence score

### If Predicted FAKE
- Queries **Google Fact Check API** with the first 12 words of the article
- Displays matching fact-check results (publisher, rating, URL)
- Generates a neutral rewrite using a HuggingFace summarization model

### URL Mode Additional Flow
```
URL Input
    │
    ├──▶ Google Safe Browsing Check (is the site malicious?)
    │
    ├──▶ trafilatura (extract article text)
    │
    └──▶ Prediction pipeline (same as text mode)
```

---

## 📊 Input Modes

| Mode | Description |
|------|-------------|
| **Paste Text** | Paste any news article or claim directly |
| **Upload CSV** | Upload a CSV with a text column for batch predictions |
| **Enter Website URL** | Enter a URL; the app extracts and analyzes the article |

---

## ⚠️ Limitations & Disclaimer

- The **suggested true rewrite** is AI-generated and may hallucinate or invent facts. Use it as a draft suggestion only.
- Model accuracy depends heavily on the quality and diversity of training data.
- The Google Fact Check API may not have matches for all claims.
- Safe Browsing only detects known malicious URLs; it does not guarantee a site is trustworthy.

---

## 🧰 Technologies Used

| Category | Tools |
|----------|-------|
| Machine Learning | Scikit-learn, XGBoost |
| NLP / Transformers | HuggingFace Transformers, NLTK |
| Web Framework | Streamlit |
| APIs | Google Fact Check Tools API, Google Safe Browsing API |
| URL Extraction | trafilatura |
| Data Handling | Pandas, NumPy |
| Model Serialization | joblib, pickle |

---

## 👤 Author

Built as a personal AI/ML project to combat online misinformation using ensemble machine learning, NLP, and real-world fact-checking APIs.
