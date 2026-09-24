import streamlit as st
import joblib
import re
import string
import math
import os
import requests
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS (kept minimal so it matches the boxed mockup look)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #0b0f1a;
    color: #e5e7eb;
}

.block-container {
    max-width: 720px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    color: #f9fafb;
    margin-bottom: 2px;
}

.main-subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 14px;
    margin-bottom: 26px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #f3f4f6;
    margin: 26px 0 10px 0;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 6px;
}

div.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 16px;
    font-weight: 700;
}
div.stButton > button:hover { filter: brightness(1.1); }

textarea {
    background: #0f1523 !important;
    color: #f9fafb !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
}

.result-box {
    border: 1px solid #374151;
    border-radius: 14px;
    padding: 20px;
    margin-top: 8px;
}
.result-box.fake { border-color: rgba(248,113,113,0.5); background: rgba(248,113,113,0.06); }
.result-box.real { border-color: rgba(52,211,153,0.5); background: rgba(52,211,153,0.06); }

.result-line { font-size: 15px; margin: 4px 0; color: #d1d5db; }
.result-line b { color: #f9fafb; }

.evidence-box {
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-size: 14px;
    color: #d1d5db;
}
.evidence-box b { color: #f9fafb; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING
# ============================================================
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

FACT_CHECK_API_KEY = st.secrets["FACT_CHECK_API_KEY"]

# Publishers you consider trustworthy — matched case-insensitively as a substring
# against the publisher name returned by the API. Add/remove as you like.
TRUSTED_PUBLISHERS = [
    "boom", "vishvas news", "factly", "india today", "the quint",
    "alt news", "politifact", "reuters", "afp fact check", "snopes",
    "factcheck.org", "the hindu", "times of india", "pib fact check",
]

def is_trusted(publisher: str) -> bool:
    p = publisher.lower()
    return any(name in p for name in TRUSTED_PUBLISHERS)

@st.cache_resource(show_spinner=False)
def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, tfidf = load_model()
MODEL_READY = model is not None and tfidf is not None

# ============================================================
# TEXT CLEANING
# ============================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================
# GOOGLE FACT CHECKING
# ============================================================
def search_fact_checks(claim):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": claim,
        "languageCode": "en",
        "pageSize": 10,
        "key": FACT_CHECK_API_KEY,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        return response.json().get("claims", [])
    except Exception:
        return []

def fact_check_news(news):
    claim = news.strip()
    if len(claim) > 500:
        claim = claim[:500]

    results = search_fact_checks(claim)
    if not results:
        return {"status": "UNVERIFIED", "reason": "No matching fact-check was found.", "evidence": []}

    evidence = []
    for item in results:
        claim_text = item.get("text", "")
        for review in item.get("claimReview", []):
            publisher = review.get("publisher", {})
            evidence.append({
                "claim": claim_text,
                "publisher": publisher.get("name", "Unknown"),
                "rating": review.get("textualRating", "Unknown"),
                "review_title": review.get("title", ""),
                "url": review.get("url", ""),
            })

    if not evidence:
        return {"status": "UNVERIFIED", "reason": "No usable fact-check evidence was found.", "evidence": []}

    # Trusted publishers first, so the most credible sources surface at the top
    evidence.sort(key=lambda item: not is_trusted(item["publisher"]))

    false_words = ["false", "fake", "incorrect", "misleading", "wrong", "pants on fire"]
    true_words = ["true", "correct", "accurate"]

    for item in evidence:
        if any(w in item["rating"].lower() for w in false_words):
            return {"status": "FALSE", "reason": "A fact-checking source rated this claim as false or misleading.", "evidence": evidence}

    for item in evidence:
        if any(w in item["rating"].lower() for w in true_words):
            return {"status": "VERIFIED", "reason": "A fact-checking source rated this claim as true or accurate.", "evidence": evidence}

    return {"status": "UNVERIFIED", "reason": "Fact-check results were found, but the rating was inconclusive.", "evidence": evidence}

def get_probabilities(transformed_text):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(transformed_text)[0]
        real_p, fake_p = probs[0] * 100, probs[1] * 100
    else:
        score = model.decision_function(transformed_text)[0]
        fake_p = (1 / (1 + math.exp(-score))) * 100
        real_p = 100 - fake_p
    prediction = model.predict(transformed_text)[0]
    return real_p, fake_p, prediction

# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Enter a news claim to check if it is real or fake</div>', unsafe_allow_html=True)

if not MODEL_READY:
    st.error(f"Model files not found. Expected `{MODEL_PATH}` and `{VECTORIZER_PATH}` in the app folder.")

# ============================================================
# SINGLE INPUT BOX (matches mockup: "Enter News / Claim")
# ============================================================
claim_text = st.text_area(
    "Enter News / Claim",
    placeholder="The government announced a new ...",
    height=140,
)

check_button = st.button("CHECK NEWS", use_container_width=True)

# ============================================================
# PREDICTION + FACT CHECK
# ============================================================
if check_button:
    if not MODEL_READY:
        st.error("The model files could not be loaded, so a prediction can't be made.")
    elif claim_text.strip() == "":
        st.warning("Please enter a news claim.")
    else:
        with st.spinner("Analyzing..."):
            cleaned_text = clean_text(claim_text)
            transformed_text = tfidf.transform([cleaned_text])
            real_probability, fake_probability, prediction = get_probabilities(transformed_text)

            is_fake_ml = prediction == 1
            ml_result = "Fake News" if is_fake_ml else "Real News"
            confidence = fake_probability if is_fake_ml else real_probability

        with st.spinner("Checking fact-checking sources..."):
            fact_result = fact_check_news(claim_text.strip())

        status = fact_result["status"]

        # ---------------- FINAL VERDICT ----------------
        # Fact-check evidence overrides the ML model's style-based guess whenever
        # a fact-checker has actually rated the claim.
        overridden = False
        if status == "FALSE":
            result = "Fake News"
            overridden = ml_result != "Fake News"
        elif status == "VERIFIED":
            result = "Real News"
            overridden = ml_result != "Real News"
        else:
            result = ml_result

        is_fake = result == "Fake News"
        card_class = "fake" if is_fake else "real"

        st.session_state.history.append({
            "title": claim_text[:30],
            "label": result,
            "confidence": confidence,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

        # ---------------- ML PREDICTION ----------------
        st.markdown('<div class="section-title">🤖 ML PREDICTION</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-box {card_class}">
                <div class="result-line"><b>Result:</b> {result}</div>
                <div class="result-line"><b>Confidence:</b> {confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if overridden:
            st.caption(f"⚠️ Adjusted from the raw ML prediction ({ml_result}) based on fact-check evidence below.")

        # ---------------- FACT CHECK ----------------
        st.markdown('<div class="section-title">🔎 FACT CHECK</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-line"><b>Status:</b> {status}</div>
                <div class="result-line"><b>Reason:</b> {fact_result["reason"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------- EVIDENCE / SOURCES ----------------
        st.markdown('<div class="section-title">📚 EVIDENCE / SOURCES</div>', unsafe_allow_html=True)

        if fact_result["evidence"]:
            for item in fact_result["evidence"]:
                badge = " ✅ Trusted Source" if is_trusted(item["publisher"]) else ""
                st.markdown(
                    f"""
                    <div class="evidence-box">
                        <b>Publisher:</b> {item["publisher"]}{badge}<br>
                        <b>Rating:</b> {item["rating"]}<br>
                        <b>Review:</b> {item["review_title"]}<br>
                        <b>Source URL:</b> {item["url"] if item["url"] else "N/A"}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No existing fact-check evidence was found for this claim.")

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div style="text-align:center; color:#6b7280; font-size:13px; margin-top:40px; padding-top:16px; border-top:1px solid #1f2937;">
    Fake News Detection System &nbsp;|&nbsp; Machine Learning Mini Project
    </div>
    """,
    unsafe_allow_html=True,
)
