📰 Fake News Detection System

📌 Project Title

Fake News Detection System Using Machine Learning

---

📖 Project Overview

The Fake News Detection System is a machine learning-based application designed to identify whether a news article is real or fake.

The system analyzes the textual content of news articles and predicts their authenticity using Natural Language Processing (NLP) and a machine learning classification model.

This application helps users quickly verify news articles and reduce the spread of misinformation.

---

🎯 Problem Statement

In today's digital world, fake news spreads rapidly through social media and online platforms. It becomes difficult for users to verify whether a news article is real or fake.

The goal of this project is to develop a machine learning-based system that automatically analyzes news content and predicts whether it is real or fake.

---

⚙️ Features

✔ Single News Prediction
Users can enter a news article and check whether it is real or fake.

✔ URL News Detection
Users can paste a news article URL and the system extracts the text and predicts its authenticity.

✔ Batch CSV Prediction
Users can upload a CSV file containing multiple news articles and the system predicts them at once.

---

🛠 Technologies Used

Programming Language

* Python

Libraries

* Pandas
* Scikit-learn
* BeautifulSoup
* Requests
* Joblib

Machine Learning Techniques

* TF-IDF Vectorization
* Multinomial Naive Bayes Classifier

Framework

* Streamlit (for web interface)

🧠 Machine Learning Workflow

1. Load Fake and Real news datasets
2. Clean and preprocess text data
3. Balance the dataset
4. Convert text into numerical features using TF-IDF
5. Train the model using Multinomial Naive Bayes
6. Evaluate the model performance
7. Save the trained model using Joblib
8. Use the saved model for predictions in the Streamlit application

---

📂 Project Structure

FakeNewsDetectionSystem

backend.py – Model training and prediction logic
app.py – Streamlit web application
fake_news_model.pkl – Trained machine learning model
tfidf_vectorizer.pkl – TF-IDF vectorizer
Fake.csv – Fake news dataset
True.csv – Real news dataset
requirements.txt – Required Python libraries
README.md – Project documentation

 🚀 How to Run the Project

Step 1: Clone the repository

git clone (your GitHub repository link)

Step 2: Install required libraries

pip install -r requirements.txt

Step 3: Run the Streamlit application

streamlit run app.py

Step 4: Open the browser and test the application.

🎥 Demo Video

https://drive.google.com/drive/folders/1-JZbD0kScUw9OmAKSuTFVhcwNod09-N_?usp=drive_link

📦 Model Files

Due to GitHub file size limitations, the trained model files are stored in Google Drive.

Download them here:
https://drive.google.com/drive/folders/1xyKHuBo83VmPWV2W9s7wwGG0jdoH3Yuv?usp=drive_link

After downloading, place the files inside the project folder before running the application.

Required files:

* fake_news_model.pkl
* tfidf_vectorizer.pkl

Conclusion

This project demonstrates how machine learning and natural language processing techniques can be used to automatically detect fake news by analyzing textual patterns in news articles.

The system provides an easy-to-use interface for users to verify news authenticity quickly.

Author

k.sri naga durga
