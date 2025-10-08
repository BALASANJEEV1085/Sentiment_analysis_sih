import os
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_supervised(X_train, y_train):
    """
    Train a TF-IDF + LogisticRegression pipeline.
    Returns: fitted pipeline
    """
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=30000)),
        ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_model(pipe, X_test, y_test):
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return acc, report, preds

def save_model(pipe, name="sentiment_model.pkl"):
    path = os.path.join(MODEL_DIR, name)
    with open(path, 'wb') as f:
        pickle.dump(pipe, f)
    return path

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

######### VADER fallback #########
_analyzer = None
def get_vader():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer

def vader_sentiment_label(text, pos_thresh=0.05, neg_thresh=-0.05):
    sid = get_vader()
    score = sid.polarity_scores(text)['compound']
    if score >= pos_thresh:
        return 'positive'
    elif score <= neg_thresh:
        return 'negative'
    else:
        return 'neutral'

def vader_sentiment_score(text):
    sid = get_vader()
    return sid.polarity_scores(text)
