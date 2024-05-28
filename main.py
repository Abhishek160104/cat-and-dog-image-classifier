import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1', error_bad_lines=False)

# Rename the columns
df = df.rename(columns={'v2': 'messages', 'v1': 'label'})

# Define a function to clean the text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove stopwords
    text = " ".join(word for word in text.split() if word not in set(stopwords.words('english')))
    return text

# Clean the text messages
df['clean_text'] = df['messages'].apply(clean_text)

# Drop rows with missing values
df = df.dropna()

# Split the data into training and testing sets
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define a function to classify the model
def classify(model, X, y):
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Classification Report:")
    print(classification_report(y, y_pred))

# Train and classify using different models
models = [LogisticRegression(), MultinomialNB(), SVC(C=3), RandomForestClassifier()]
for model in models:
    print("Model:", model.__class__.__name__)
    classify(model, X_train_tfidf, y_train)
    print()