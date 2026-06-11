
import streamlit as st
from joblib import load
import re
import numpy as np

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="centered"
)

# ---------------- DARK / LIGHT MODE ----------------

theme = st.sidebar.selectbox(
    "🎨 Choose Theme",
    ["Dark Mode", "Light Mode"]
)

if theme == "Dark Mode":

    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }

        textarea {
            background-color: #262730 !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
        <style>
        .stApp {
            background-color: #F5F7FA;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

vectorizer = load("tfidf_vectorizer.joblib")
model = load("linear_svm_model.joblib")
# ---------------- TEXT CLEANING ----------------

def clean_text(text):

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text.lower()

# ---------------- TITLE ----------------

st.markdown(
    "<h1 style='text-align:center;'>🐦 Twitter Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;'>Analyze Tweets Using Machine Learning 🚀</h4>",
    unsafe_allow_html=True
)

st.write("")

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Options")

show_cleaned = st.sidebar.checkbox("Show Cleaned Tweet")

show_probability = st.sidebar.checkbox("Show Confidence Score")

# ---------------- INPUT ----------------

tweet = st.text_area(
    "✍️ Enter Tweet",
    placeholder="Type your tweet here..."
)

# ---------------- BUTTON ----------------

if st.button("🔍 Analyze Sentiment"):

    if tweet.strip() == "":
        st.warning("⚠️ Please enter a tweet.")

    else:

        cleaned_tweet = clean_text(tweet)

        tweet_vector = vectorizer.transform([cleaned_tweet])

        prediction = model.predict(tweet_vector)

        # ---------------- OPTIONAL INFO ----------------

        if show_cleaned:
            st.info(f"🧹 Cleaned Tweet: {cleaned_tweet}")

        # ---------------- RESULT ----------------

        st.write("")
        st.subheader("📊 Prediction Result")

        if prediction[0] == 0:

            st.error("😡 Negative Tweet")

        else:

            st.success("😊 Positive Tweet")
            st.balloons()

        # ---------------- CONFIDENCE SCORE ----------------

        try:

            probability = np.max(model.predict_proba(tweet_vector))

            if show_probability:

                st.progress(float(probability))

                st.write(
                    f"🎯 Confidence Score: {probability * 100:.2f}%"
                )

        except:
            pass

# ---------------- FOOTER ----------------

st.write("")
st.markdown("---")

st.markdown(
    "<center>Made with ❤️ using Streamlit & Machine Learning</center>",
    unsafe_allow_html=True
)

