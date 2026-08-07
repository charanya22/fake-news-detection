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
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#5A7A9A;
    font-size:18px;
    margin-bottom:20px;
}

.stButton>button{
    width:100%;
    background-color:#2E6DA4;
    color:white;
    border:none;
    border-radius:10px;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#245785;
    color:white;
}

.footer{
    text-align:center;
    color:#7A7A7A;
    font-size:14px;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

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
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, vectorizer = load_model()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ℹ About")

    st.write(
        "This application predicts whether a news article is **Real** or **Fake** "
        "using **Machine Learning** and **Natural Language Processing (NLP)**."
    )

    st.divider()

    st.subheader("🛠 Technologies")

    st.markdown("""
- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
""")

    st.divider()

    st.subheader("👩‍💻 Developer")
    st.write("**K. S. Charanya**")

# ---------------- Header ----------------
st.markdown(
    '<div class="title">📰 Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze news articles using Machine Learning and Natural Language Processing (NLP).</div>',
    unsafe_allow_html=True
)

st.info(
    "💡 Enter a complete news article below and click **Analyze Article** to predict whether it is **Real** or **Fake**."
)

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
        transformed = vectorizer.transform([news])
        prediction = model.predict(transformed)[0]
        confidence = model.predict_proba(transformed).max() * 100

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success(f"✅ This appears to be REAL News\n\nConfidence: {confidence:.2f}%")
        else:
            st.error(f"❌ This appears to be FAKE News\n\nConfidence: {confidence:.2f}%")

        st.progress(int(confidence))

        if confidence >= 90:
            st.balloons()

        st.divider()

# ---------------- Footer ----------------
st.markdown(
    '<div class="footer">Developed by <b>K. S. Charanya</b> • Fake News Detection using Machine Learning</div>',
    unsafe_allow_html=True
)