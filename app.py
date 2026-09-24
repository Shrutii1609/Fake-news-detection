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
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Main Background ---------- */
    .stApp {
        background: radial-gradient(circle at 15% 0%, #151a2e 0%, #0b0f1a 45%, #05070d 100%);
        color: #e5e7eb;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10131f 0%, #0b0e18 100%);
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f9fafb;
    }

    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li {
        color: #9ca3af;
    }

    /* ---------- Main Container ---------- */
    .block-container {
        max-width: 1200px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Header ---------- */
    .hero {
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 6px;
    }

    .hero-badge {
        font-size: 34px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        width: 60px;
        height: 60px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 24px rgba(99,102,241,0.35);
    }

    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #f9fafb;
        letter-spacing: -1px;
        margin: 0;
    }

    .main-subtitle {
        font-size: 16px;
        color: #9ca3af;
        margin: 4px 0 30px 78px;
    }

    /* ---------- Section Titles ---------- */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #f3f4f6;
        margin: 30px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ---------- Card wrapper for inputs ---------- */
    .card-panel {
        background: rgba(17, 24, 39, 0.65);
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 26px;
        backdrop-filter: blur(6px);
    }

    /* ---------- Labels ---------- */
    label, .stTextInput label, .stTextArea label {
        color: #d1d5db !important;
        font-weight: 500 !important;
    }

    /* ---------- Text Inputs ---------- */
    div[data-baseweb="input"] {
        background: #0f1523;
        border: 1px solid #374151;
        border-radius: 10px;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.3);
    }

    input { color: #f9fafb !important; }

    textarea {
        background: #0f1523 !important;
        color: #f9fafb !important;
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
    }

    textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.3) !important;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-size: 16px;
        font-weight: 600;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        filter: brightness(1.1);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(99,102,241,0.35);
    }

    /* ---------- Result Card ---------- */
    .result-card {
        border-radius: 20px;
        padding: 32px;
        margin-top: 10px;
        text-align: center;
        border: 1px solid #374151;
        animation: fadeIn 0.4s ease;
    }

    .result-card.fake {
        background: radial-gradient(circle at top, rgba(248,113,113,0.12), rgba(17,24,39,0.9));
        border-color: rgba(248,113,113,0.35);
    }

    .result-card.real {
        background: radial-gradient(circle at top, rgba(52,211,153,0.12), rgba(17,24,39,0.9));
        border-color: rgba(52,211,153,0.35);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-label {
        color: #9ca3af;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .result-value {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .confidence-text {
        color: #9ca3af;
        font-size: 14px;
    }

    /* ---------- Stat / Metric Cards ---------- */
    .stat-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        height: 100%;
        transition: 0.2s ease;
    }

    .stat-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
    }

    .stat-number {
        font-size: 24px;
        font-weight: 700;
        color: #f9fafb;
    }

    .stat-label {
        font-size: 12.5px;
        color: #9ca3af;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ---------- Info Box ---------- */
    .info-box {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px 18px;
        margin-top: 14px;
        color: #9ca3af;
        line-height: 1.6;
        font-size: 14px;
    }

    /* ---------- Badge pill ---------- */
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(99,102,241,0.15);
        color: #a5b4fc;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* ---------- History list ---------- */
    .history-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        border-radius: 10px;
        background: #111827;
        border: 1px solid #1f2937;
        margin-bottom: 8px;
        font-size: 13px;
        color: #d1d5db;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #1f2937;
    }

    /* Hide default streamlit chrome for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING (robust: won't crash the whole app if missing)
# ============================================================

MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
# ============================================================
# GOOGLE FACT CHECK API
# ============================================================

FACT_CHECK_API_KEY = st.secrets["FACT_CHECK_API_KEY"]


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
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)      # URLs
    text = re.sub(r"<.*?>", "", text)                         # HTML tags
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    text = re.sub(r"\d+", "", text)                           # numbers
    text = re.sub(r"\s+", " ", text).strip()                  # extra spaces
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
        "key": FACT_CHECK_API_KEY
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return []

        data = response.json()

        return data.get("claims", [])

    except Exception:
        return []


def fact_check_news(news):

    claim = news.strip()

    if len(claim) > 500:
        claim = claim[:500]

    results = search_fact_checks(claim)

    if not results:
        return {
            "status": "UNVERIFIED",
            "reason": "No matching fact-check was found.",
            "evidence": []
        }

    evidence = []

    for item in results:

        claim_text = item.get("text", "")

        reviews = item.get("claimReview", [])

        for review in reviews:

            publisher = review.get("publisher", {})

            evidence.append({
                "claim": claim_text,
                "publisher": publisher.get(
                    "name",
                    "Unknown"
                ),
                "rating": review.get(
                    "textualRating",
                    "Unknown"
                ),
                "review_title": review.get(
                    "title",
                    ""
                ),
                "url": review.get(
                    "url",
                    ""
                )
            })

    if not evidence:
        return {
            "status": "UNVERIFIED",
            "reason": "No usable fact-check evidence was found.",
            "evidence": []
        }

    false_words = [
        "false",
        "fake",
        "incorrect",
        "misleading",
        "wrong",
        "pants on fire"
    ]

    true_words = [
        "true",
        "correct",
        "accurate"
    ]

    for item in evidence:

        rating = item["rating"].lower()

        if any(word in rating for word in false_words):

            return {
                "status": "FALSE",
                "reason": "A fact-checking source rated this claim as false or misleading.",
                "evidence": evidence
            }

    for item in evidence:

        rating = item["rating"].lower()

        if any(word in rating for word in true_words):

            return {
                "status": "VERIFIED",
                "reason": "A fact-checking source rated this claim as true or accurate.",
                "evidence": evidence
            }

    return {
        "status": "UNVERIFIED",
        "reason": "Fact-check results were found, but the rating was inconclusive.",
        "evidence": evidence
    }

def get_probabilities(transformed_text):
    """Returns (real_probability, fake_probability, prediction) as percentages / label."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(transformed_text)[0]
        real_p, fake_p = probs[0] * 100, probs[1] * 100
    else:
        score = model.decision_function(transformed_text)[0]
        fake_p = (1 / (1 + math.exp(-score))) * 100
        real_p = 100 - fake_p

    prediction = model.predict(transformed_text)[0]
    return real_p, fake_p, prediction


def make_gauge_svg(confidence: float, is_fake: bool) -> str:
    """Builds a pure SVG donut/gauge — no external charting library needed."""
    color = "#f87171" if is_fake else "#34d399"
    radius = 70
    stroke = 16
    circumference = 2 * math.pi * radius
    offset = circumference * (1 - confidence / 100)

    svg = f"""
    <div style="display:flex; justify-content:center; align-items:center; padding:10px 0;">
        <svg width="200" height="200" viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="{radius}" fill="none"
                    stroke="#1f2937" stroke-width="{stroke}"/>
            <circle cx="100" cy="100" r="{radius}" fill="none"
                    stroke="{color}" stroke-width="{stroke}"
                    stroke-linecap="round"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"
                    transform="rotate(-90 100 100)"
                    style="transition: stroke-dashoffset 0.6s ease;"/>
            <text x="100" y="94" text-anchor="middle"
                  font-size="30" font-weight="700" fill="#f9fafb"
                  font-family="Inter, sans-serif">{confidence:.0f}%</text>
            <text x="100" y="118" text-anchor="middle"
                  font-size="12" fill="#9ca3af"
                  font-family="Inter, sans-serif">CONFIDENCE</text>
        </svg>
    </div>
    """
    return svg


# ============================================================
# SESSION STATE (prediction history)
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:10px 0 22px 0;">
            <div style="font-size:42px;">📰</div>
            <h2 style="margin-bottom:2px;">Fake News Detection</h2>
            <p style="font-size:13px;">Machine Learning Based System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Model Status")
    if MODEL_READY:
        st.success("Model & vectorizer loaded")
    else:
        st.error("Model files not found")
        st.caption(f"Expected `{MODEL_PATH}` and `{VECTORIZER_PATH}` in the app folder.")

    st.markdown("---")
    st.markdown("### About")
    st.write(
        "This app uses Machine Learning to classify news as **Real** or **Fake** "
"and checks available fact-checking sources for additional evidence."
    )

    st.markdown("---")
    st.markdown("### Technology")
    st.markdown(
        '<span class="pill">Python</span>'
        '<span class="pill">Scikit-learn</span>'
        '<span class="pill">TF-IDF</span>'
        '<span class="pill">Streamlit</span>'
        '<span class="pill">Plotly</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Recent Predictions")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            badge = "🔴" if item["label"] == "FAKE NEWS" else "🟢"
            st.markdown(
                f'<div class="history-item"><span>{badge} {item["title"][:22]}...</span>'
                f'<span>{item["confidence"]:.0f}%</span></div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions yet.")

    st.markdown("---")
    st.caption("Fake News Detection • ML Mini Project")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">📰</div>
        <div class="main-title">Fake News Detection</div>
    </div>
    <div class="main-subtitle">
        Analyze a news article and predict whether it is Real or Fake using Machine Learning.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NEWS INPUT
# ============================================================

st.markdown('<div class="section-title">🔍 Enter News Details</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)

    title = st.text_input(
        "News Title",
        placeholder="Enter the headline of the news article...",
    )

    article = st.text_area(
        "News Article",
        placeholder="Paste the complete news article here...",
        height=220,
    )

    col_btn, col_clear = st.columns([4, 1])
    with col_btn:
        predict_button = st.button("🚀 Predict News", use_container_width=True)
    with col_clear:
        clear_button = st.button("Clear", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

if clear_button:
    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if not MODEL_READY:
        st.error(
            "The model files could not be loaded, so a prediction can't be made. "
            "Please make sure `fake_news_model.pkl` and `tfidf_vectorizer.pkl` are "
            "in the same folder as this app."
        )

    elif title.strip() == "" and article.strip() == "":
        st.warning("Please enter a news title or article.")

    else:
        with st.spinner("Analyzing the article..."):
            combined_text = f"{title} {article}"
            cleaned_text = clean_text(combined_text)
            transformed_text = tfidf.transform([cleaned_text])
            real_probability, fake_probability, prediction = get_probabilities(transformed_text)

        is_fake = prediction == 1
        result = "FAKE NEWS" if is_fake else "REAL NEWS"
        confidence = fake_probability if is_fake else real_probability
        card_class = "fake" if is_fake else "real"
        value_color = "#f87171" if is_fake else "#34d399"

        # Save to history
        st.session_state.history.append({
            "title": title if title.strip() else article[:30],
            "label": result,
            "confidence": confidence,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

        # ---------------- Result card ----------------
        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-label">Prediction Result</div>
                <div class="result-value" style="color:{value_color};">{result}</div>
                <div class="confidence-text">Confidence Score: {confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------- Gauge + probability breakdown ----------------
        st.markdown('<div class="section-title">📊 Prediction Details</div>', unsafe_allow_html=True)

        col_gauge, col_bars = st.columns([1, 1.4])

        with col_gauge:
            st.markdown(make_gauge_svg(confidence, is_fake), unsafe_allow_html=True)

        with col_bars:
            st.markdown(
                f"""
                <div class="stat-card" style="margin-bottom:12px; text-align:left;">
                    <div class="stat-label">Real News Probability</div>
                    <div class="stat-number">{real_probability:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(int(real_probability))

            st.markdown(
                f"""
                <div class="stat-card" style="margin:12px 0; text-align:left;">
                    <div class="stat-label">Fake News Probability</div>
                    <div class="stat-number">{fake_probability:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(int(fake_probability))

        # ---------------- Article statistics ----------------
        st.markdown('<div class="section-title">📈 Article Statistics</div>', unsafe_allow_html=True)

        word_count = len(combined_text.split())
        character_count = len(combined_text)
        title_count = len(title.split())
        article_count = len(article.split())

        stats = [
            ("Total Words", word_count),
            ("Characters", character_count),
            ("Title Words", title_count),
            ("Article Words", article_count),
        ]

        cols = st.columns(4)
        for col, (label, value) in zip(cols, stats):
            with col:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-number">{value}</div>
                        <div class="stat-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        # ====================================================
        # FACT CHECKING
        # ====================================================

        st.markdown(
            '<div class="section-title">🔎 Fact Checking</div>',
            unsafe_allow_html=True
        )

        # Use title as the main claim
        if title.strip():
            fact_check_claim = title.strip()
        else:
            fact_check_claim = article.strip()[:500]

        with st.spinner("Checking fact-checking sources..."):

            fact_result = fact_check_news(fact_check_claim)

        status = fact_result["status"]

        # Display status
        if status == "VERIFIED":
            st.success("✅ VERIFIED")

        elif status == "FALSE":
            st.error("❌ FALSE / MISLEADING")

        else:
            st.warning("⚠️ UNVERIFIED")

        # Display reason
        st.markdown(
            f"""
            <div class="info-box">
                <b>Fact-Check Status:</b> {status}<br><br>
                <b>Reason:</b> {fact_result["reason"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display evidence
        if fact_result["evidence"]:

            st.markdown(
                '<div class="section-title">📚 Fact-Check Evidence</div>',
                unsafe_allow_html=True
            )

            for i, item in enumerate(
                fact_result["evidence"],
                1
            ):

                st.markdown(
                    f"""
                    <div class="stat-card"
                         style="text-align:left; margin-bottom:15px;">

                        <div class="stat-label">
                            SOURCE {i}
                        </div>

                        <div style="
                            font-size:18px;
                            font-weight:600;
                            margin:8px 0;
                        ">
                            {item["publisher"]}
                        </div>

                        <div style="
                            color:#a5b4fc;
                            margin-bottom:8px;
                        ">
                            Rating: {item["rating"]}
                        </div>

                        <div style="
                            color:#d1d5db;
                            margin-bottom:10px;
                        ">
                            {item["review_title"]}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if item["url"]:
                    st.link_button(
                        "🔗 View Fact Check Source",
                        item["url"]
                    )

        else:

            st.info(
                "No existing fact-check was found for this claim. "
                "The claim should be treated as unverified."
            )
    


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Fake News Detection System &nbsp;|&nbsp; Machine Learning Mini Project
    </div>
    """,
    unsafe_allow_html=True,
)
