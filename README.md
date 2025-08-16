# 🌐 Language Detection using Machine Learning & NLP

A real-time multilingual language detection system that uses classical machine learning algorithms and TF-IDF vectorization to identify the language of a given text input. This project supports multiple languages and evaluates different models to select the best-performing one.

# 🧾 Overview

This project leverages Natural Language Processing (NLP) and Machine Learning (ML) to build a language detection model. Given a text input, the system predicts the language using the most accurate classifier from a tested set.

# 📚 Dataset

- **Source:** [Aman Kharwal's dataset on GitHub](https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv)
- **Languages Supported:** Includes texts in English, French, Spanish, German, Italian, Dutch, Hindi, and more.
- **Structure:** 
  - `Text`: The sentence or phrase.
  - `Language`: The corresponding language label.

# 🧠 Models Implemented

- Multinomial Naive Bayes
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)

All models are trained and evaluated, with the best model selected based on accuracy.

 # ⚙️ How It Works

1. Load and preprocess the dataset.
2. Apply TF-IDF vectorization to the text data.
3. Split data into training and testing sets.
4. Train five different ML models:
  Multinomial Naive Bayes

  Logistic Regression

  Random Forest Classifier

  K-Nearest Neighbors

  Support Vector Classifier

5. Evaluate and select the best-performing model.
6. Enter text in real-time to detect the language using the best model.

# 💻 Sample Output

```bash
Enter a text to detect its language (or type 'exit' to quit): Je suis ravi de vous rencontrer

Prediction Result:
╒═══════════════════╕
│ Detected Language │
╞═══════════════════╡
│        French      │
╘═══════════════════╛
