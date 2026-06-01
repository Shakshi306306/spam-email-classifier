import pandas as pd
import numpy as np
import nltk
import string
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Text preprocessing function
def clean_text(text):

    text = text.lower()

    text = ''.join([
        char for char in text
        if char not in string.punctuation
    ])

    words = text.split()

    words = [
        word for word in words
        if word not in stopwords.words('english')
    ]

    return ' '.join(words)

# Apply cleaning
df['message'] = df['message'].apply(clean_text)

# Convert text to vectors
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['message'])

y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, 'spam_model.pkl')

# Save vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model Saved Successfully")

