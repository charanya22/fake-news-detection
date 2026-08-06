import streamlit as st
import pickle

# -------------------- Load Model --------------------
with open("models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -------------------- Title --------------------
st.title("📰 Fake News Detection System")
st.markdown("### Detect whether a news article is **Real** or **Fake** using Machine Learning.")

st.divider()

# -------------------- Input --------------------
news = st.text_area(
    "Paste the news article below",
    height=250,
    placeholder="Paste the complete news article here..."
)

# -------------------- Button --------------------
if st.button("Predict", use_container_width=True):

    if news.strip() == "":
        st.warning("⚠ Please enter a news article.")
    else:

        with st.spinner("Analyzing the article..."):

            transformed = vectorizer.transform([news])
            prediction = model.predict(transformed)
            probability = model.predict_proba(transformed)

        st.divider()

        confidence = max(probability[0]) * 100

        if prediction[0] == 1:
            st.success("✅ This appears to be REAL News.")
        else:
            st.error("❌ This appears to be FAKE News.")

        st.metric("Confidence", f"{confidence:.2f}%")

st.divider()

st.markdown("### About this Project")

st.info(
    """
This project uses **Natural Language Processing (NLP)** and
**Machine Learning (Logistic Regression + TF-IDF)** to classify
news articles as **Real** or **Fake**.
"""
)