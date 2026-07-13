import streamlit as st
import joblib

# Load Model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Amazon Review Sentiment Analyzer")

st.write("Enter a product review and click Predict.")

review = st.text_area(
    "Review",
    height=200,
    placeholder="Write your review..."
)

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        vector = vectorizer.transform([review])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        confidence = max(probability) * 100

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.write(f"Confidence : {confidence:.2f}%")

st.sidebar.title("About")

st.sidebar.write("""
Dataset:
Amazon Reviews

Model:
Logistic Regression

Feature Extraction:
TF-IDF
""")
