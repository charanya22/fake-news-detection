import streamlit as st
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

.stApp{
    background-color:#F8FAFC;
}

.title{
    text-align:center;
    color:#1E4B7A;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#5A7A9A;
    font-size:18px;
    margin-bottom:20px;
}

.stTextArea textarea{
    border-radius:12px;
    border:1px solid #C5D9EA;
}

.stButton>button{
    width:100%;
    background-color:#2E6DA4;
    color:white;
    border:none;
    border-radius:12px;
    font-weight:bold;
    height:50px;
}

.stButton>button:hover{
    background-color:#245785;
}

.result-real{
    background:#E6F7ED;
    border-left:6px solid #22A559;
    border-radius:12px;
    padding:20px;
}

.result-fake{
    background:#FDECEC;
    border-left:6px solid #E5484D;
    border-radius:12px;
    padding:20px;
}

.result-label{
    font-size:24px;
    font-weight:bold;
}

.confidence-text{
    font-size:16px;
    color:#555;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    try:
        with open("models/fake_news_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, vectorizer = load_model()
# ---------------- Header ----------------
st.markdown(
    "<div class='title'>📰 Fake News Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Analyze news articles using Machine Learning and Natural Language Processing (NLP).</div>",
    unsafe_allow_html=True
)

st.info("💡 Enter a complete news article below and click **Analyze Article** to predict whether it is Real or Fake.")

# ---------------- Input ----------------
left, center, right = st.columns([0.5, 7, 0.5])

with center:
    news = st.text_area(
        "📄 Paste News Article",
        height=250,
        placeholder="Paste a complete news article here..."
    )

with center:
    analyze = st.button("🔎 Analyze Article")

# ---------------- Prediction ----------------
if analyze:

    if news.strip() == "":
        st.warning("⚠ Please enter a news article.")

    else:

        st.subheader("📊 Prediction Result")

        transformed = vectorizer.transform([news])
        prediction = model.predict(transformed)[0]
        confidence = model.predict_proba(transformed).max() * 100

        if prediction == 1:

            st.markdown(f"""
            <div class="result-real">
                <div class="result-label">✅ This appears to be REAL News</div>
                <div class="confidence-text">
                    Confidence: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

        else:

            st.markdown(f"""
            <div class="result-fake">
                <div class="result-label">❌ This appears to be FAKE News</div>
                <div class="confidence-text">
                    Confidence: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

# ---------------- Footer ----------------
st.divider()

st.markdown(
    "<div class='footer'>Developed by K. S. Charanya | Fake News Detection using Machine Learning</div>",
    unsafe_allow_html=True
)