#!/usr/bin/env python3
"""
Ultra-Fast Fake News Detector
Minimal dependencies, fast execution
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import json

print("START TRAINING")

# Load data
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# Sample
true_sample = true_df.head(3000)
fake_sample = fake_df.head(3000)

title_col = 'title' if 'title' in true_df.columns else 0
text_col = 'text' if 'text' in true_df.columns else 1

# Create texts
texts = []
labels = []

for _, row in true_sample.iterrows():
    texts.append(str(row[title_col]) + " " + str(row[text_col]))
    labels.append(1)

for _, row in fake_sample.iterrows():
    texts.append(str(row[title_col]) + " " + str(row[text_col]))
    labels.append(0)

print(f"Loaded {len(texts)} articles")

# Vectorize
vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
X = vectorizer.fit_transform(texts)
print(f"Vectorized: {X.shape}")

# Train
model = LogisticRegression(max_iter=100)
model.fit(X, labels)
print(f"Trained")

# Evaluate
score = model.score(X, labels)
print(f"Accuracy: {score:.2%}")

# Save
with open('model_ultra.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer_ultra.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('metadata_ultra.json', 'w') as f:
    json.dump({"accuracy": score}, f)

print("TRAINING COMPLETE")
print(f"Saved: model_ultra.pkl, vectorizer_ultra.pkl")
