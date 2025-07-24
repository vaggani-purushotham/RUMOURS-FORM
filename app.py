import joblib
import pickle
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load trained model and vectorizer
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
model_path = 'trained_model.pkl'  # Path to your trained model file
vectorizer_path = 'tfidf_vectorizer.pkl'  # Path to your TF-IDF vectorizer file
rfc = joblib.load('rfc_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Add lexicon-based features
def add_lexicon_features(text):
    compound_score = sia.polarity_scores(text)['compound']
    return compound_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned_text = clean_text(text)
    lexicon_score = add_lexicon_features(text)
    tfidf_features = tfidf_vectorizer.transform([cleaned_text])
    combined_features = pd.concat([pd.DataFrame(tfidf_features.toarray()), pd.DataFrame([lexicon_score])], axis=1)
    prediction = rfc.predict(combined_features)
    return render_template('result.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
