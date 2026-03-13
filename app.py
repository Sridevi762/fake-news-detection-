import streamlit as st
import joblib
import pandas as pd

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# ===============================
# Load Model (Cached for Speed)
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# Header
# ===============================
st.title("📰 Fake News Detection System")
st.caption("AI-based system to verify news authenticity")

tab1, tab2 = st.tabs(["🔍 Single News", "📂 Batch CSV"])

# ===============================
# SINGLE NEWS
# ===============================
with tab1:

    news = st.text_area("Paste news content here", height=200)

    if st.button("Analyze News"):

        if news.strip() == "":
            st.warning("Please enter some news text.")
        else:
            # Transform
            input_vector = vectorizer.transform([news])

            # Predict
            prediction = model.predict(input_vector)[0]
            probability = model.predict_proba(input_vector)[0]

            confidence = max(probability) * 100

            st.subheader("Result")

            if prediction == 1:
                st.success("✅ REAL NEWS")
            else:
                st.error("❌ FAKE NEWS")

            st.write(f"**Confidence:** {confidence:.2f}%")

# ===============================
# BATCH CSV
# ===============================
with tab2:

    file = st.file_uploader("Upload CSV (must contain column name: text)", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("CSV must contain 'text' column.")
        else:
            vectors = vectorizer.transform(df["text"])
            predictions = model.predict(vectors)

            df["prediction"] = predictions
            df["prediction_label"] = df["prediction"].map({1: "REAL NEWS", 0: "FAKE NEWS"})

            st.dataframe(df, use_container_width=True)

            chart = df["prediction_label"].value_counts()
            st.bar_chart(chart)

            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Results",
                csv,
                "results.csv",
                "text/csv"
            )

# ===============================
# Footer
# ===============================
st.markdown(
    "<center style='color:gray;'>© 2026 Fake News Detection System</center>",
    unsafe_allow_html=True
)