# Fake-news-detection

## 📌 Project Overview

Fake news spreads quickly through social media and online platforms. 
This project is a Machine Learning based system that helps identify whether a news article is **Real or Fake**.

The system uses **TF-IDF** for text feature extraction and **Logistic Regression** for classification.

It also includes a **Fact Checking feature** that uses the Google Fact Check Tools API to find available fact-checking information for a given claim.

---

## 🎯 Objectives

- Detect whether a news article is Real or Fake.
- Use Machine Learning for automatic news classification.
- Extract useful text features using TF-IDF.
- Provide prediction confidence.
- Check available fact-checking information from external sources.
- Display fact-checking evidence and source links through a Streamlit interface.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF
- Logistic Regression
- Joblib
- Streamlit
- Google Fact Check Tools API

---

## 🧠 Machine Learning Algorithm

### Logistic Regression

Logistic Regression is used as the main classification algorithm.

The model classifies news into two categories:

- **Real News → 0**
- **Fake News → 1**

The news text is first converted into numerical features using **TF-IDF**, which are then given to the Logistic Regression model.

---

## 🔄 System Workflow

```text
User enters News
        ↓
Text Cleaning
        ↓
TF-IDF Feature Extraction
        ↓
Logistic Regression
        ↓
Real / Fake Prediction
        ↓
Confidence Score
        ↓
Google Fact Check API
        ↓
Available Fact-Check Evidence
        ↓
Verified / False / Unverified
