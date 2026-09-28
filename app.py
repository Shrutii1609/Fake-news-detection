import streamlit as st
import joblib
import re
import string
import os
import math
from datetime import datetime
from collections import Counter

import pandas as pd
import textstat

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="TruthLens AI | Fake News Detection",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0b0f1a;
    color: #e5e7eb;
}
.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #f9fafb;
    margin-bottom: 4px;
}
.main-subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 14px;
    margin-bottom: 26px;
}
.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #f3f4f6;
    margin: 28px 0 12px 0;
    border-bottom: 1px solid #293244;
    padding-bottom: 8px;
}
div.stButton > button {
    width: 100%;
    min-height: 46px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 15px;
    font-weight: 700;
}
div.stButton > button:hover {
    filter: brightness(1.1);
    color: white;
}
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
    background: rgba(31, 41, 55, 0.18);
}
.result-box.fake {
    border-color: rgba(248,113,113,0.65);
    background: rgba(248,113,113,0.07);
}
.result-box.real {
    border-color: rgba(52,211,153,0.65);
    background: rgba(52,211,153,0.07);
}
.result-line {
    font-size: 15px;
    margin: 5px 0;
    color: #d1d5db;
    overflow-wrap: anywhere;
}
.result-line b {
    color: #f9fafb;
}
.small-note {
    color: #9ca3af;
    font-size: 12px;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING
# ============================================================
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model and TF-IDF vectorizer from disk."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None

    try:
        trained_model = joblib.load(MODEL_PATH)
        trained_vectorizer = joblib.load(VECTORIZER_PATH)
        return trained_model, trained_vectorizer
    except Exception as exc:
        st.error(f"Could not load model files: {exc}")
        return None, None


model, tfidf = load_model()
MODEL_READY = model is not None and tfidf is not None

# ============================================================
# TEXT CLEANING
# IMPORTANT: Keep this preprocessing consistent with training.
# If your training pipeline used a different clean_text function,
# replace this function with the exact same preprocessing.
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
# FEATURE 3: ARTICLE QUALITY & READABILITY ANALYSIS
# ============================================================
SENSATIONAL_WORDS = [
    "shocking",
    "unbelievable",
    "breaking",
    "exposed",
    "secret",
    "scandal",
    "explosive",
    "miracle",
    "stunning",
    "bombshell",
    "must see",
    "you won't believe",
    "you will not believe",
    "jaw-dropping",
    "urgent",
    "terrifying",
    "disaster",
    "outrage",
]


def analyze_article_quality(text: str) -> dict:
    """Calculate writing-style and readability statistics."""

    original_text = text.strip()
    words = re.findall(r"\b[\w'-]+\b", original_text)
    word_count = len(words)

    # Count non-empty sentence-like segments.
    sentence_parts = re.split(r"[.!?]+", original_text)
    sentences = [part.strip() for part in sentence_parts if part.strip()]
    sentence_count = len(sentences)

    avg_sentence_length = (
        word_count / sentence_count if sentence_count else 0.0
    )

    # Flesch Reading Ease is most meaningful for English text.
    try:
        readability_score = (
            textstat.flesch_reading_ease(original_text)
            if word_count >= 3 else None
        )
    except Exception:
        readability_score = None

    if readability_score is None:
        readability_level = "Not enough text"
    elif readability_score >= 70:
        readability_level = "Easy to read"
    elif readability_score >= 50:
        readability_level = "Moderately difficult"
    elif readability_score >= 30:
        readability_level = "Difficult to read"
    else:
        readability_level = "Very difficult to read"

    lower_text = original_text.lower()
    found_phrases = []

    for phrase in SENSATIONAL_WORDS:
        pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
        matches = re.findall(pattern, lower_text)
        found_phrases.extend([phrase] * len(matches))

    # Uppercase percentage excludes digits, spaces, and punctuation.
    alphabetic_chars = [char for char in original_text if char.isalpha()]
    uppercase_chars = [char for char in alphabetic_chars if char.isupper()]
    uppercase_percentage = (
        len(uppercase_chars) / len(alphabetic_chars) * 100
        if alphabetic_chars else 0.0
    )

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "readability_score": readability_score,
        "readability_level": readability_level,
        "sensational_words": found_phrases,
        "sensational_count": len(found_phrases),
        "uppercase_percentage": uppercase_percentage,
        "exclamation_count": original_text.count("!"),
        "question_count": original_text.count("?"),
    }


def display_article_quality(quality: dict):
    """Render article-quality metrics and writing-style indicators."""

    st.markdown(
        '<div class="section-title">📝 ARTICLE QUALITY & READABILITY</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Word Count", quality["word_count"])
    col2.metric("Sentences", quality["sentence_count"])
    col3.metric(
        "Avg. Sentence Length",
        f'{quality["avg_sentence_length"]:.1f} words',
    )

    col4, col5, col6 = st.columns(3)
    score = quality["readability_score"]
    col4.metric(
        "Readability Score",
        f"{score:.1f}" if score is not None else "N/A",
    )
    col5.metric("Sensational Phrases", quality["sensational_count"])
    col6.metric(
        "Uppercase Letters",
        f'{quality["uppercase_percentage"]:.1f}%',
    )

    st.caption(
        f'Readability interpretation: {quality["readability_level"]}. '
        "Flesch Reading Ease is a writing readability estimate."
    )

    st.markdown("**Punctuation Analysis**")
    punct1, punct2 = st.columns(2)
    punct1.metric("Exclamation Marks (!)", quality["exclamation_count"])
    punct2.metric("Question Marks (?)", quality["question_count"])

    st.markdown("**Sensational Words / Phrases Found**")
    if quality["sensational_words"]:
        phrase_counts = Counter(quality["sensational_words"])
        for phrase, count in phrase_counts.items():
            st.warning(f'"{phrase}" — detected {count} time(s)')
    else:
        st.success("No phrases from the built-in sensational-language list were detected.")

    st.info(
        "These indicators describe writing style only. Sensational wording, "
        "readability, capitalization, or punctuation does not prove that a "
        "news article is real or fake."
    )


# ============================================================
# PREDICTION PROBABILITIES AND LABEL MAPPING
# ============================================================
def get_class_probability(transformed_text, class_label):
    """Return probability for a specific class label, if available."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(transformed_text)[0]
    classes = list(getattr(model, "classes_", []))

    try:
        class_index = classes.index(class_label)
        return float(probabilities[class_index]) * 100
    except (ValueError, IndexError):
        return None


def get_prediction(transformed_text):
    """
    Predict using the model's actual class labels.

    Assumption for this project: label 0 = Real and label 1 = Fake.
    Confirm this mapping against the labels used when training the model.
    """
    predicted_label = model.predict(transformed_text)[0]

    # Existing app's expected dataset mapping:
    # class 1 -> Fake News; class 0 -> Real News.
    # This mapping must match the training dataset.
    fake_label = 1
    real_label = 0

    if predicted_label == fake_label:
        result = "Fake News"
    elif predicted_label == real_label:
        result = "Real News"
    else:
        result = f"Class {predicted_label}"

    fake_probability = get_class_probability(transformed_text, fake_label)
    real_probability = get_class_probability(transformed_text, real_label)

    # Fallback for estimators without predict_proba.
    if fake_probability is None or real_probability is None:
        if hasattr(model, "decision_function"):
            decision = model.decision_function(transformed_text)
            decision_value = float(decision[0])

            # Binary classifier decision scores correspond to classes_[1].
            classes = list(getattr(model, "classes_", []))
            positive_class = classes[1] if len(classes) > 1 else fake_label

            # Numerically stable sigmoid.
            if decision_value >= 0:
                sigmoid = 1 / (1 + math.exp(-decision_value))
            else:
                exp_value = math.exp(decision_value)
                sigmoid = exp_value / (1 + exp_value)

            positive_probability = sigmoid * 100

            if positive_class == fake_label:
                fake_probability = positive_probability
                real_probability = 100 - positive_probability
            else:
                real_probability = positive_probability
                fake_probability = 100 - positive_probability
        else:
            # Do not invent confidence if the model exposes no probability
            # or decision score.
            fake_probability = None
            real_probability = None

    if result == "Fake News":
        confidence = fake_probability
    elif result == "Real News":
        confidence = real_probability
    else:
        confidence = None

    return (
        result,
        real_probability,
        fake_probability,
        confidence,
        predicted_label,
    )


# ============================================================
# FEATURE 5: PREDICTION HISTORY & ANALYTICS DASHBOARD
# ============================================================
def display_analytics_dashboard():
    st.markdown(
        '<div class="section-title">📊 PREDICTION HISTORY & ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    history = st.session_state.get("history", [])

    if not history:
        st.info(
            "No predictions yet. Analyze a news article to start building "
            "your dashboard. History is stored for this Streamlit session."
        )
        return

    df = pd.DataFrame(history)

    total = len(df)
    fake_count = int((df["label"] == "Fake News").sum())
    real_count = int((df["label"] == "Real News").sum())

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Total Analyzed", total)
    metric2.metric("Fake Predictions", fake_count)
    metric3.metric("Real Predictions", real_count)

    st.markdown("### Prediction Distribution")
    chart_data = pd.DataFrame(
        {
            "Prediction": ["Fake News", "Real News"],
            "Count": [fake_count, real_count],
        }
    ).set_index("Prediction")
    st.bar_chart(chart_data, use_container_width=True)

    # Confidence trend only includes valid numeric confidence values.
    if "confidence" in df.columns:
        confidence_df = df.copy()
        confidence_df["confidence"] = pd.to_numeric(
            confidence_df["confidence"], errors="coerce"
        )
        confidence_df = confidence_df.dropna(subset=["confidence"])

        if not confidence_df.empty:
            st.markdown("### Prediction Confidence")
            confidence_df = confidence_df.reset_index(drop=True)
            confidence_df["Analysis Number"] = range(
                1, len(confidence_df) + 1
            )
            confidence_chart = confidence_df.set_index("Analysis Number")[
                ["confidence"]
            ].rename(columns={"confidence": "Confidence (%)"})
            st.line_chart(confidence_chart, use_container_width=True)

    st.markdown("### Recent Predictions")

    display_columns = [
        column
        for column in ["title", "label", "confidence", "time"]
        if column in df.columns
    ]
    history_display = df[display_columns].copy()

    rename_map = {
        "title": "Article",
        "label": "Prediction",
        "confidence": "Confidence (%)",
        "time": "Date & Time",
    }
    history_display = history_display.rename(columns=rename_map)

    if "Confidence (%)" in history_display.columns:
        history_display["Confidence (%)"] = history_display[
            "Confidence (%)"
        ].apply(
            lambda value: (
                f"{float(value):.2f}%"
                if pd.notna(value)
                else "Unavailable"
            )
        )

    st.dataframe(
        history_display.iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "History is currently stored in Streamlit session state. "
        "It may be cleared when the session ends or the app restarts."
    )

    if st.button("Clear Prediction History", key="clear_history"):
        st.session_state.history = []
        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">📰 TruthLens AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-subtitle">'
    'Analyze news with machine learning and explore its writing style'
    '</div>',
    unsafe_allow_html=True,
)

if not MODEL_READY:
    st.error(
        f"Model files not found or could not be loaded. Expected "
        f"`{MODEL_PATH}` and `{VECTORIZER_PATH}` in the app folder. "
        "Place both files beside app.py and restart the app."
    )

# ============================================================
# NEWS INPUT
# ============================================================
claim_text = st.text_area(
    "Enter News / Claim",
    placeholder="Paste a news article or headline here...",
    height=180,
    max_chars=20000,
    help="Enter the headline or article text you want the model to analyze.",
)

character_count = len(claim_text)
st.caption(f"{character_count:,} / 20,000 characters")

check_button = st.button("ANALYZE NEWS", use_container_width=True)

# ============================================================
# PREDICTION + ARTICLE QUALITY ANALYSIS
# ============================================================
if check_button:
    if not MODEL_READY:
        st.error(
            "The model files could not be loaded, so a prediction cannot be made."
        )
    elif not claim_text.strip():
        st.warning("Please enter a news article or claim.")
    else:
        cleaned_text = clean_text(claim_text)

        if not cleaned_text:
            st.warning(
                "The text is empty after preprocessing. Please enter meaningful "
                "words, not only punctuation, numbers, or URLs."
            )
        else:
            with st.spinner("Analyzing the news text..."):
                try:
                    transformed_text = tfidf.transform([cleaned_text])

                    (
                        ml_result,
                        real_probability,
                        fake_probability,
                        confidence,
                        predicted_label,
                    ) = get_prediction(transformed_text)

                    quality = analyze_article_quality(claim_text)

                except Exception as exc:
                    st.error(f"An error occurred during analysis: {exc}")
                    st.stop()

            # Save prediction in session history.
            st.session_state.history.append(
                {
                    "title": claim_text.strip().replace("\n", " ")[:100],
                    "label": ml_result,
                    "confidence": confidence,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            # ---------------- ML PREDICTION ----------------
            st.markdown(
                '<div class="section-title">🤖 ML PREDICTION</div>',
                unsafe_allow_html=True,
            )

            card_class = (
                "fake" if ml_result == "Fake News" else "real"
            )

            confidence_text = (
                f"{confidence:.2f}%"
                if confidence is not None
                else "Unavailable"
            )

            st.markdown(
                f"""
                <div class="result-box {card_class}">
                    <div class="result-line"><b>Prediction:</b> {ml_result}</div>
                    <div class="result-line"><b>Model confidence:</b> {confidence_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display both class probabilities where available.
            if real_probability is not None and fake_probability is not None:
                prob_col1, prob_col2 = st.columns(2)
                prob_col1.metric("Real-class probability", f"{real_probability:.2f}%")
                prob_col2.metric("Fake-class probability", f"{fake_probability:.2f}%")

                st.progress(
                    min(max(float(fake_probability) / 100, 0.0), 1.0),
                    text=f"Fake-class probability: {fake_probability:.2f}%",
                )
            else:
                st.info(
                    "This model does not expose class probabilities. "
                    "The predicted class is shown, but a confidence percentage "
                    "is not available."
                )

            st.caption(
                "The model prediction reflects patterns learned from its training "
                "dataset. Model confidence is not a guarantee of factual accuracy "
                "and is not independent fact verification."
            )

            # ---------------- ARTICLE QUALITY ----------------
            display_article_quality(quality)

# ============================================================
# ANALYTICS DASHBOARD
# Always rendered so users can view session history between analyses.
# ============================================================
st.divider()
display_analytics_dashboard()

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div style="text-align:center; color:#6b7280; font-size:13px;
                margin-top:40px; padding-top:16px;
                border-top:1px solid #1f2937;">
        TruthLens AI &nbsp;|&nbsp; Machine Learning & NLP Project
    </div>
    """,
    unsafe_allow_html=True,
)
