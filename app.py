import streamlit as st
import pandas as pd

# Import functions from backend
from backend import load_model, predict_single, predict_from_url, predict_batch

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# ===============================
# LOAD MODEL
# ===============================
model, vectorizer = load_model()

# ===============================
# HEADER
# ===============================
st.title("📰 Fake News Detection System")
st.caption("AI-based system to verify news authenticity")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Single News",
    "🌐 URL News",
    "📂 Batch CSV"
])

# ===============================
# SINGLE NEWS PREDICTION
# ===============================
with tab1:

    news = st.text_area("Paste news content here", height=200)

    if st.button("Analyze News"):

        if news.strip() == "":
            st.warning("Please enter some news text.")

        else:
            label, confidence = predict_single(news, model, vectorizer)

            st.subheader("Result")

            if "REAL" in label:
                st.success(label)
            else:
                st.error(label)

            st.write(f"Confidence: {confidence}%")

# ===============================
# URL NEWS PREDICTION
# ===============================
with tab2:

    url = st.text_input("Enter news article URL")

    if st.button("Analyze URL"):

        if url.strip() == "":
            st.warning("Please enter a URL")

        else:
            label, confidence = predict_from_url(url, model, vectorizer)

            st.subheader("Result")

            if "REAL" in label:
                st.success(label)
            elif "FAKE" in label:
                st.error(label)
            else:
                st.warning(label)

            if confidence != 0:
                st.write(f"Confidence: {confidence}%")

# ===============================
# BATCH CSV PREDICTION
# ===============================
with tab3:

    file = st.file_uploader(
        "Upload CSV file (must contain column 'text')",
        type=["csv"]
    )

    if file:

        df = pd.read_csv(file)

        try:
            results = predict_batch(file, model, vectorizer)

            st.dataframe(results)

            chart = results["prediction_label"].value_counts()
            st.bar_chart(chart)

            csv = results.to_csv(index=False)

            st.download_button(
                "Download Results",
                csv,
                "results.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(str(e))

# ===============================
# FOOTER
# ===============================
st.markdown(
    "<center style='color:gray;'>© 2026 Fake News Detection System</center>",
    unsafe_allow_html=True
)
