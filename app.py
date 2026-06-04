# app.py - Fake News Detector + Google Fact Check (claims:search) + SafeBrowsing + trafilatura
import os
import io
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from time import sleep
from transformers import pipeline
from typing import List, Optional
import re
import requests
import trafilatura

# Try to detect GPU for transformer pipelines (optional)
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

st.set_page_config(page_title="Fake News Detector (FactCheck)", layout="wide")

# -----------------------
# Helper: ProbVotingClassifier (so unpickling works across files)
# -----------------------
class ProbVotingClassifier:
    def __init__(self, pipes: List[object], weights: Optional[List[float]] = None):
        self.pipes = pipes
        if weights is None:
            self.weights = [1.0] * len(pipes)
        else:
            self.weights = weights

    def predict_proba(self, X):
        probs = None
        total_w = 0.0
        for p, w in zip(self.pipes, self.weights):
            pr = p.predict_proba(X)
            if probs is None:
                probs = w * pr
            else:
                probs = probs + w * pr
            total_w += w
        return probs / total_w

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# -----------------------
# Modern cached loader for models (Streamlit)
# -----------------------
@st.cache_resource()
def load_models():
    progress_container = st.empty()
    prog = progress_container.progress(0)

    prog.progress(5)
    try:
        pipe_lr = joblib.load("models/pipe_lr.joblib")
        prog.progress(25)
    except Exception as e:
        raise RuntimeError(f"Failed to load models/pipe_lr.joblib: {e}")

    try:
        pipe_xgb = joblib.load("models/pipe_xgb.joblib")
        prog.progress(45)
    except Exception as e:
        raise RuntimeError(f"Failed to load models/pipe_xgb.joblib: {e}")

    ensemble = None
    try:
        if os.path.exists("models/ensemble.joblib"):
            ensemble = joblib.load("models/ensemble.joblib")
            prog.progress(65)
    except Exception:
        ensemble = None

    if ensemble is None:
        ensemble = ProbVotingClassifier([pipe_lr, pipe_xgb], weights=[1.0, 1.0])
        prog.progress(70)

    sentiment_pipe = None
    try:
        if os.path.exists("models/sentiment_model_name.pkl"):
            with open("models/sentiment_model_name.pkl", "rb") as f:
                sent_name = pickle.load(f)
            device = 0 if HAS_CUDA else -1
            sentiment_pipe = pipeline("sentiment-analysis", model=sent_name, device=device)
            prog.progress(85)
    except Exception:
        sentiment_pipe = None

    gen_pipe = None
    try:
        if os.path.exists("models/gen_model_name.pkl"):
            with open("models/gen_model_name.pkl", "rb") as f:
                gen_name = pickle.load(f)
            device = 0 if HAS_CUDA else -1
            gen_pipe = pipeline("summarization", model=gen_name, framework="pt", device=device)
            prog.progress(100)
    except Exception:
        gen_pipe = None

    sleep(0.15)
    progress_container.empty()
    return pipe_lr, pipe_xgb, ensemble, sentiment_pipe, gen_pipe

# Load models once (cached)
try:
    pipe_lr, pipe_xgb, ensemble, sentiment_pipe, gen_pipe = load_models()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

# -----------------------
# Safe transformer helpers (handle long inputs)
# -----------------------
def _get_pipe_max_tokens(pipe, fallback=512):
    try:
        tok = pipe.tokenizer
        m = getattr(tok, "model_max_length", None) or getattr(tok, "max_len", None)
        if m is None or m > 10000:
            return fallback
        return int(m)
    except Exception:
        return fallback

def safe_sentiment(text: str, pipe):
    if pipe is None:
        return None
    max_len = _get_pipe_max_tokens(pipe)
    try:
        out = pipe(text, truncation=True, max_length=max_len)
        if isinstance(out, list) and len(out) > 0:
            return out[0]
        return out
    except Exception:
        truncated = text[:4000]
        try:
            out = pipe(truncated, truncation=True, max_length=max_len)
            return out[0] if isinstance(out, list) else out
        except Exception:
            return None

def _chunk_text_by_sentences(text, tokenizer, max_tokens):
    import re
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sents:
        if not current:
            current = sent
            continue
        try:
            length = len(tokenizer.encode(current + " " + sent, add_special_tokens=False))
        except Exception:
            length = len((current + " " + sent).split())
        if length <= max_tokens:
            current = (current + " " + sent).strip()
        else:
            chunks.append(current.strip())
            try:
                sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
            except Exception:
                sent_len = len(sent.split())
            if sent_len > max_tokens:
                approx_chars = max(200, int(len(sent) * max_tokens / max(sent_len,1)))
                start = 0
                while start < len(sent):
                    piece = sent[start:start+approx_chars].strip()
                    if piece:
                        chunks.append(piece)
                    start += approx_chars
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current.strip())
    return chunks

def safe_summarize(text: str, pipe, max_chunk_summary_len=120, min_chunk_summary_len=30):
    if pipe is None:
        return None

    tok_max = _get_pipe_max_tokens(pipe, fallback=512)
    max_input_tokens = max(128, min(1024, tok_max) - 32)
    tokenizer = pipe.tokenizer

    chunks = _chunk_text_by_sentences(text, tokenizer, max_input_tokens)
    if not chunks:
        return ""

    chunk_summaries = []
    prog_bar = st.progress(0)
    n = len(chunks)
    for i, ch in enumerate(chunks):
        try:
            out = pipe(ch, max_length=max_chunk_summary_len, min_length=min_chunk_summary_len, truncation=True, do_sample=False)
            if isinstance(out, list) and len(out) > 0:
                chunk_summaries.append(out[0].get('summary_text') or out[0].get('generated_text') or "")
            elif isinstance(out, dict):
                chunk_summaries.append(out.get('summary_text',''))
            else:
                chunk_summaries.append(str(out))
        except Exception:
            chunk_summaries.append(ch[:max_chunk_summary_len*4])
        prog_bar.progress(int((i+1)/n * 100))

    prog_bar.empty()

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = " ".join(chunk_summaries)
    try:
        final = pipe(combined, max_length=160, min_length=30, truncation=True, do_sample=False)
        if isinstance(final, list) and len(final) > 0:
            return final[0].get('summary_text') or final[0].get('generated_text') or ""
        elif isinstance(final, dict):
            return final.get('summary_text','')
        else:
            return str(final)
    except Exception:
        return combined[:1000]

# -------------------------------
# New: trafilatura extractor & Google APIs (Safe Browsing + Fact Check)
# -------------------------------

# Replace these placeholders with your private keys (do NOT paste in public chat)
GOOGLE_SAFE_BROWSING_KEY = "AIzaSyDo5Pauh0U-xk31riiNVNVGATwxSVxmY1k"
GOOGLE_FACTCHECK_KEY = "AIzaSyDo5Pauh0U-xk31riiNVNVGATwxSVxmY1k"

SAFE_BROWSING_ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def extract_text_from_url(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False)
            if extracted and len(extracted) > 50:
                return extracted
        return None
    except Exception:
        return None

def check_website_safety(url):
    payload = {
        "client": {"clientId": "fake-news-app", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE","SOCIAL_ENGINEERING","UNWANTED_SOFTWARE","POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        }
    }
    params = {"key": GOOGLE_SAFE_BROWSING_KEY}
    try:
        r = requests.post(SAFE_BROWSING_ENDPOINT, params=params, json=payload, timeout=10)
        data = r.json()
        if "matches" in data:
            return False, data["matches"]
        return True, None
    except Exception as e:
        return None, str(e)

def query_factcheck_claims(query_text, language_code="en"):
    """
    Query Google Fact Check tools (claims:search).
    Returns list of dicts with keys: ['text','claimReview' list ...] or None
    """
    params = {
        "query": " ".join(query_text.split()[:12]),  # limit length
        "key": GOOGLE_FACTCHECK_KEY,
        "languageCode": language_code
    }
    try:
        r = requests.get(FACTCHECK_ENDPOINT, params=params, timeout=10)
        data = r.json()
        # debug print (streamlit logs will show)
        print("FactCheck response keys:", list(data.keys()) if isinstance(data, dict) else type(data))
        if "claims" not in data:
            return None
        claims = data["claims"]
        results = []
        for c in claims:
            # collect useful fields
            claim_text = c.get("text","")
            claimant = c.get("claimant","")
            claim_date = c.get("claimDate","")
            # claimReview is a list (may be absent)
            claim_reviews = []
            for cr in c.get("claimReview", []):
                cr_item = {
                    "publisher_name": cr.get("publisher", {}).get("name", ""),
                    "url": cr.get("url", ""),
                    "title": cr.get("title", ""),
                    "review_date": cr.get("reviewDate",""),
                    "textual_rating": cr.get("textualRating",""),
                    "language": cr.get("languageCode","")
                }
                claim_reviews.append(cr_item)
            results.append({
                "claim_text": claim_text,
                "claimant": claimant,
                "claim_date": claim_date,
                "claim_reviews": claim_reviews
            })
        return results if results else None
    except Exception as e:
        print("FactCheck API error:", e)
        return None

# -----------------------
# UI
# -----------------------
st.title("📡 Fake News Detector Using Machine Learning and NLP")

st.markdown(
    """
    **What this app does**
    - Predicts whether input text/article is likely **FAKE** or **TRUE** using your ensemble model.
    - If **TRUE**, displays sentiment (if sentiment model present).
    - If **FAKE**, queries Google Fact Check (claims:search) to find matching fact-checks and displays them.
    - Supports pasting text, uploading a CSV, or entering a website URL (Safe Browsing check + extraction).
    """
)

st.sidebar.header("Settings")
show_raw = st.sidebar.checkbox("Show raw predictions (for CSV)", value=True)
device_info = "GPU" if HAS_CUDA else "CPU"
st.sidebar.write(f"Transformer device: **{device_info}**")

mode = st.radio("Input mode", ("Paste text", "Upload CSV (column: text)", "Enter Website URL"))

# Single-text mode
if mode == "Paste text":
    text = st.text_area("Paste the news text / article here", height=300)
    if st.button("Predict"):
        if not text.strip():
            st.warning("Please paste some text.")
        else:
            with st.spinner("Predicting..."):
                try:
                    pred = ensemble.predict([text])[0]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred = None

                if pred is None:
                    st.error("Prediction failed; check model logs.")
                elif pred == 0:
                    st.success("✅ Model prediction: TRUE news")
                    if sentiment_pipe is not None:
                        with st.spinner("Running sentiment analysis..."):
                            sent = safe_sentiment(text, sentiment_pipe)
                            if sent:
                                label = sent.get('label') or sent.get('score')
                                score = sent.get('score', None)
                                st.write(f"**Sentiment:** {label} — confidence {score if score is not None else 'N/A'}")
                            else:
                                st.warning("Sentiment model failed.")
                    else:
                        st.info("Sentiment model not available.")
                else:
                    st.error("❗ Model prediction: FAKE news")
                    st.info("Searching Google Fact Check for matching claims...")
                    fc_results = query_factcheck_claims(text)
                    if fc_results:
                        st.markdown("**Fact-check matches (top results):**")
                        for i, r in enumerate(fc_results, start=1):
                            st.markdown(f"**Match {i} — Claim:** {r.get('claim_text')}")
                            if r.get("claim_reviews"):
                                for cr in r["claim_reviews"]:
                                    st.write(f"- Publisher: **{cr['publisher_name']}**")
                                    st.write(f"  - Rating: {cr.get('textual_rating')}")
                                    st.write(f"  - Title: {cr.get('title')}")
                                    st.write(f"  - URL: {cr.get('url')}")
                                    st.write("---")
                            else:
                                st.write("_No claim reviews found for this claim._")
                    else:
                        st.info("No Google Fact Check matches found.")
                        # fallback behavior: generator & optional external searches (not included here)
                    if gen_pipe is not None:
                        with st.spinner("Generating suggested true-news rewrite..."):
                            summary = safe_summarize(text, gen_pipe)
                            if summary:
                                st.subheader("Suggested 'true' rewrite")
                                st.write(summary)
                            else:
                                st.warning("Generator model failed or not available.")
                    else:
                        st.info("Generator model not available.")

# CSV mode
elif mode == "Upload CSV (column: text)":
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
        st.write("Columns:", df.columns.tolist())
        text_col = st.selectbox("Select text column", options=df.columns.tolist())
        if st.button("Run predictions on CSV"):
            texts = df[text_col].astype(str).tolist()
            results = []
            progress_bar = st.progress(0)
            n = len(texts)
            for i, t in enumerate(texts):
                try:
                    p = ensemble.predict([t])[0]
                except Exception:
                    p = None
                row = {"text": t, "pred": "ERROR" if p is None else ("TRUE" if p == 0 else "FAKE")}
                if p == 0:
                    if sentiment_pipe is not None:
                        sent = safe_sentiment(t, sentiment_pipe)
                        if sent:
                            row["sentiment"] = sent.get('label')
                            row["sent_conf"] = float(sent.get('score', 0))
                        else:
                            row["sentiment"] = ""
                            row["sent_conf"] = ""
                    else:
                        row["sentiment"] = ""
                        row["sent_conf"] = ""
                    row["suggested_true"] = ""
                    row["factcheck_matches"] = ""
                elif p == 1:
                    # try fact check
                    fc = query_factcheck_claims(t)
                    if fc:
                        row["factcheck_matches"] = json.dumps(fc)
                    else:
                        row["factcheck_matches"] = ""
                    if gen_pipe is not None:
                        try:
                            summary = safe_summarize(t, gen_pipe)
                            row["suggested_true"] = summary
                        except Exception:
                            row["suggested_true"] = ""
                    else:
                        row["suggested_true"] = ""
                    row["sentiment"] = ""
                    row["sent_conf"] = ""
                else:
                    row["sentiment"] = ""
                    row["sent_conf"] = ""
                    row["suggested_true"] = ""
                    row["factcheck_matches"] = ""
                results.append(row)
                progress_bar.progress(int((i+1)/n * 100))
            progress_bar.empty()
            out_df = pd.DataFrame(results)
            st.write("Predictions (first 20 rows):")
            st.write(out_df.head(20))
            if show_raw:
                st.download_button("Download predictions CSV", out_df.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")

# URL mode
else:
    url = st.text_input("Enter website/article URL to check:")
    if st.button("Check URL"):
        if not url.strip():
            st.warning("Please paste a URL.")
        else:
            st.info("Checking website safety (Google Safe Browsing)...")
            safe, info = check_website_safety(url)
            if safe is False:
                st.error("Website flagged as unsafe by Google Safe Browsing:")
                st.write(info)
                st.stop()
            elif safe is None:
                st.warning(f"Could not determine safety: {info}")

            st.info("Extracting article text from the URL...")
            extracted = extract_text_from_url(url)
            if not extracted:
                st.error("Could not extract article content from the URL.")
                st.stop()

            st.success("Article extracted successfully.")
            st.write("### Article preview:")
            st.write(extracted[:800] + "..." if len(extracted) > 800 else extracted)

            with st.spinner("Predicting..."):
                try:
                    pred = ensemble.predict([extracted])[0]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred = None

                if pred is None:
                    st.error("Prediction failed; check model logs.")
                elif pred == 0:
                    st.success("✅ Model prediction: TRUE news")
                    if sentiment_pipe is not None:
                        with st.spinner("Running sentiment analysis..."):
                            sent = safe_sentiment(extracted, sentiment_pipe)
                            if sent:
                                label = sent.get('label') or sent.get('score')
                                score = sent.get('score', None)
                                st.write(f"**Sentiment:** {label} — confidence {score if score is not None else 'N/A'}")
                            else:
                                st.warning("Sentiment model failed.")
                    else:
                        st.info("Sentiment model not available.")
                else:
                    st.error("❗ Model prediction: FAKE news")
                    st.info("Searching Google Fact Check for matching claims...")
                    fc_results = query_factcheck_claims(extracted)
                    if fc_results:
                        st.markdown("**Fact-check matches (top results):**")
                        for i, r in enumerate(fc_results, start=1):
                            st.markdown(f"**Match {i} — Claim:** {r.get('claim_text')}")
                            if r.get("claim_reviews"):
                                for cr in r["claim_reviews"]:
                                    st.write(f"- Publisher: **{cr['publisher_name']}**")
                                    st.write(f"  - Rating: {cr.get('textual_rating')}")
                                    st.write(f"  - Title: {cr.get('title')}")
                                    st.write(f"  - URL: {cr.get('url')}")
                                    st.write("---")
                            else:
                                st.write("_No claim reviews found for this claim._")
                    else:
                        st.info("No Google Fact Check matches found.")
                        # optionally continue to generator
                    if gen_pipe is not None:
                        with st.spinner("Generating suggested true-news rewrite..."):
                            summary = safe_summarize(extracted, gen_pipe)
                            if summary:
                                st.subheader("Suggested 'true' rewrite")
                                st.write(summary)
                            else:
                                st.warning("Generator model failed or not available.")
                    else:
                        st.info("Generator model not available.")

# Footer notes
st.markdown("---")
st.caption(
    "⚠️ The 'suggested true rewrite' is generated by a model and may hallucinate or invent facts. "
    "Use it as a drafted suggestion only. High accuracy depends on dataset quality and model training."
)
