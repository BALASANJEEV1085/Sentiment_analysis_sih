import re
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK assets available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def simple_clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.lower()
    # remove urls/emails
    s = re.sub(r'http\S+|www\S+|https\S+', '', s)
    s = re.sub(r'\S+@\S+', '', s)
    # remove non-alpha
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # collapse spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize_text(s):
    toks = word_tokenize(s)
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 1 and not t.isnumeric()]
    return toks

def preprocess_dataframe(df, text_column):
    """
    Adds a 'clean_text' column with preprocessed text.
    """
    df = df.copy()
    df['clean_text'] = df[text_column].astype(str).apply(simple_clean_text)
    return df

def split_train_test(df, label_column, test_size=0.2, random_state=42):
    X = df['clean_text'].values
    y = df[label_column].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
