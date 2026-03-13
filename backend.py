import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# ===============================
# 1️⃣ LOAD DATASET
# ===============================

def load_dataset(fake_path="Fake.csv", true_path="True.csv"):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0  # Fake
    true_df["label"] = 1  # Real

    df = pd.concat([fake_df, true_df], ignore_index=True)
    return df


# ===============================
# 2️⃣ CLEAN TEXT
# ===============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_dataset(df):
    df["text"] = df["text"].apply(clean_text)
    return df


# ===============================
# 3️⃣ BALANCE DATASET
# ===============================

def balance_dataset(df):
    df_fake = df[df["label"] == 0]
    df_real = df[df["label"] == 1]

    df_real_upsampled = resample(
        df_real,
        replace=True,
        n_samples=len(df_fake),
        random_state=42
    )

    df_balanced = pd.concat([df_fake, df_real_upsampled])
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
    return df_balanced


# ===============================
# 4️⃣ TRAIN MODEL
# ===============================

def train_model(df):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "fake_news_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    return model, vectorizer


# ===============================
# 5️⃣ LOAD SAVED MODEL
# ===============================

def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


# ===============================
# 6️⃣ SINGLE TEXT PREDICTION
# ===============================

def predict_single(news_text, model, vectorizer):
    cleaned = clean_text(news_text)
    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]

    confidence = max(probabilities) * 100
    label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    return label, round(confidence, 2)


# ===============================
# 7️⃣ URL TEXT EXTRACTION
# ===============================

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text

    except Exception:
        return None


# ===============================
# 8️⃣ URL PREDICTION
# ===============================

def predict_from_url(url, model, vectorizer):
    article_text = extract_text_from_url(url)

    if not article_text or len(article_text.strip()) == 0:
        return "Could not extract content from URL.", 0

    return predict_single(article_text, model, vectorizer)


# ===============================
# 9️⃣ BATCH PREDICTION
# ===============================

def predict_batch(csv_path, model, vectorizer):
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("CSV must contain 'text' column")

    df["cleaned_text"] = df["text"].apply(clean_text)

    X_batch = vectorizer.transform(df["cleaned_text"])

    df["prediction"] = model.predict(X_batch)
    df["confidence"] = model.predict_proba(X_batch).max(axis=1) * 100

    df["prediction_label"] = df["prediction"].apply(
        lambda x: "REAL" if x == 1 else "FAKE"
    )

    return df[["text", "prediction_label", "confidence"]]


# ===============================
# 🔟 MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    df = load_dataset()
    df = clean_dataset(df)
    df = balance_dataset(df)

    model, vectorizer = train_model(df)

    # Test sample
    test_news = "The government launched a new digital education policy."
    label, confidence = predict_single(test_news, model, vectorizer)
    print(f"\nPrediction: {label}, Confidence: {confidence}%")
