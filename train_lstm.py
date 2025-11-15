"""
LSTM FAKE NEWS DETECTION
========================
Based on Kaggle: satyamsss/fake-news-prediction-lstm-97-accurate
Deep Learning approach with 97%+ accuracy
"""

import pandas as pd
import numpy as np
import warnings
import pickle
from pathlib import Path
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("LSTM FAKE NEWS DETECTION")
print("="*80 + "\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("[1/6] Loading datasets...", end=" ")

fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

fake['label'] = 0
true['label'] = 1

fake['text'] = fake['title'].fillna('') + ' ' + fake['text'].fillna('')
true['text'] = true['title'].fillna('') + ' ' + true['text'].fillna('')

combined = pd.concat([fake[['text', 'label']], true[['text', 'label']]], ignore_index=True)
combined = combined[combined['text'].str.len() > 10].drop_duplicates().dropna()

print(f"OK ({len(combined)} articles)\n")

# ============================================================================
# 2. TEXT PREPROCESSING
# ============================================================================
print("[2/6] Text preprocessing...", end=" ")

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(combined['text'].values)

X = tokenizer.texts_to_sequences(combined['text'].values)
X = pad_sequences(X, maxlen=max_len, padding='post')
y = combined['label'].values

print(f"OK (Vocab: {len(tokenizer.word_index)}, Sequence length: {max_len})\n")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("[3/6] Train-test split...", end=" ")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"OK (Train: {X_train.shape[0]}, Test: {X_test.shape[0]})\n")

# ============================================================================
# 4. BUILD LSTM MODEL
# ============================================================================
print("[4/6] Building LSTM model...")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"  Model summary:")
model.summary()

# ============================================================================
# 5. TRAIN MODEL
# ============================================================================
print("\n[5/6] Training LSTM model (this will take 3-5 minutes)...\n")

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ============================================================================
# 6. EVALUATE & SAVE
# ============================================================================
print("\n[6/6] Evaluating and saving...\n")

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Test Accuracy:  {test_accuracy*100:.2f}%")
print(f"Accuracy:       {acc*100:.2f}%")
print(f"Precision:      {prec:.4f}")
print(f"Recall:         {rec:.4f}")
print(f"F1-Score:       {f1:.4f}\n")

print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Save model and tokenizer
print("\n  Saving models...", end=" ")
model.save('model_lstm.h5')
print("OK")

print("  Saving tokenizer...", end=" ")
with open('tokenizer_lstm.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("OK")

print("  Saving metadata...", end=" ")
with open('metadata_lstm.pkl', 'wb') as f:
    pickle.dump({
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'max_words': max_words,
        'max_len': max_len,
        'articles': len(combined)
    }, f)
print("OK\n")

print("="*80)
print(f"LSTM MODEL TRAINED!")
print(f"Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")
print("="*80 + "\n")

print("Files created:")
print("  - model_lstm.h5")
print("  - tokenizer_lstm.pkl")
print("  - metadata_lstm.pkl\n")
