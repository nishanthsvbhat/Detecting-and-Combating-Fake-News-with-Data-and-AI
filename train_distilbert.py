"""
Simple and Stable Fake News Detector Training
Using DistilBERT (smaller, more stable) instead of DeBERTa
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FAKE NEWS DETECTOR - DistilBERT Training")
print("=" * 80)

# Use DistilBERT - lightweight and stable
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = "./distilbert_fake_news_model"

print(f"\n[*] Using Model: {MODEL_NAME}")
print(f"[*] Output Directory: {OUTPUT_DIR}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[*] Loading data...")

try:
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')
    print(f"[+] True articles: {len(true_df)}")
    print(f"[+] Fake articles: {len(fake_df)}")
except FileNotFoundError as e:
    print(f"[-] Error: {e}")
    print("Please ensure True.csv and Fake.csv are in current directory")
    exit(1)

# Detect columns
title_col = 'title' if 'title' in true_df.columns else true_df.columns[0]
text_col = 'text' if 'text' in true_df.columns else true_df.columns[1]

# Prepare texts and labels
texts = []
labels = []

# Real news (label=1)
for idx, row in true_df.head(5000).iterrows():
    text = str(row[title_col]) + " " + str(row[text_col])
    texts.append(text[:512])
    labels.append(1)

# Fake news (label=0)
for idx, row in fake_df.head(5000).iterrows():
    text = str(row[title_col]) + " " + str(row[text_col])
    texts.append(text[:512])
    labels.append(0)

print(f"\n[+] Total samples: {len(texts)}")
print(f"    Real: {sum(labels)}")
print(f"    Fake: {len(texts) - sum(labels)}")

# ============================================================================
# LOAD MODEL AND TOKENIZER
# ============================================================================

print(f"\n[*] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"[*] Loading model...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
except:
    # If the model doesn't work with num_labels, use it as-is
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Device: {device}")
model.to(device)

# ============================================================================
# TOKENIZE DATA
# ============================================================================

print("\n[*] Tokenizing data...")

encodings = tokenizer(
    texts,
    truncation=True,
    max_length=512,
    padding=True,
    return_tensors="pt"
)

labels_tensor = torch.tensor(labels)

# Create dataset
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

dataset = FakeNewsDataset(encodings, labels_tensor)

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print(f"\n[*] Train/Test Split:")
print(f"    Train: {len(train_dataset)} samples")
print(f"    Test: {len(test_dataset)} samples")

# ============================================================================
# TRAINING
# ============================================================================

print("\n[*] Setting up training...")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
epochs = 2

print("[*] Starting training...")
print("=" * 80)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0
    batch_count = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / batch_count
    print(f"    Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"    Validation Accuracy: {accuracy:.4f} ({correct}/{total})")

# ============================================================================
# SAVE MODEL
# ============================================================================

print(f"\n[*] Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

final_accuracy = correct / total
print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
print(f"Correct Predictions: {correct}/{total}")

# ============================================================================
# TEST INFERENCE
# ============================================================================

print("\n" + "=" * 80)
print("TEST INFERENCE")
print("=" * 80)

# Create pipeline for easy inference
classifier = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

test_texts = [
    "Scientists discover breakthrough cure for cancer",
    "Fake breaking: President secretly meets with aliens in underground base",
    "Stock market shows steady growth in quarterly reports",
]

print("\nTesting predictions:")
for text in test_texts:
    try:
        result = classifier(text[:512])
        label = result[0]['label']
        score = result[0]['score']
        print(f"\n[*] Text: {text[:50]}...")
        print(f"    Label: {label}")
        print(f"    Confidence: {score:.2%}")
    except Exception as e:
        print(f"    Error: {e}")

# ============================================================================
# SAVE METADATA
# ============================================================================

metadata = {
    "model": MODEL_NAME,
    "train_samples": len(train_dataset),
    "test_samples": len(test_dataset),
    "accuracy": float(final_accuracy),
    "epochs": epochs,
    "batch_size": 16
}

import json
with open('distilbert_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 80)
print("[+] TRAINING COMPLETE!")
print(f"[+] Model saved to: {OUTPUT_DIR}")
print(f"[+] Final Accuracy: {final_accuracy:.2%}")
print("=" * 80)
