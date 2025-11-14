#!/usr/bin/env python
"""
Train RoBERTa/DeBERTa Fine-Tuned Fake News Detector on ISOT Dataset
Phase 1: Transformer Models Implementation

Usage:
  python train_transformer.py --model roberta-base --epochs 5 --batch_size 16
  python train_transformer.py --model microsoft/deberta-base --epochs 5 --batch_size 16

Output:
  - Saves best model to: models/roberta_best_epoch_X.pth
  - Evaluation metrics on test set
  - Comparison with current ensemble
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers_detector import RobertaFakeNewsDetector, DeBertaFakeNewsDetector
import torch


def load_isot_dataset(true_csv='True.csv', fake_csv='Fake.csv'):
    """Load ISOT dataset"""
    print("\n" + "="*70)
    print("LOADING ISOT FAKE NEWS DATASET")
    print("="*70)
    
    print(f"\nReading {true_csv}...")
    true_df = pd.read_csv(true_csv)
    print(f"✓ Loaded {len(true_df)} real articles")
    
    print(f"Reading {fake_csv}...")
    fake_df = pd.read_csv(fake_csv)
    print(f"✓ Loaded {len(fake_df)} fake articles")
    
    # Label: 1 = real, 0 = fake
    true_df['label'] = 1
    fake_df['label'] = 0
    
    # Combine
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create text column (title + text for richer content)
    df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['text'] = df['text'].str.strip()
    
    print(f"\n{'─'*70}")
    print(f"Total samples: {len(df)}")
    print(f"Real (label=1): {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"Fake (label=0): {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"{'─'*70}\n")
    
    return df['text'].tolist(), df['label'].tolist()


def main():
    parser = argparse.ArgumentParser(description='Train RoBERTa/DeBERTa Fake News Detector')
    parser.add_argument('--model', type=str, default='roberta-base',
                        help='Model name (roberta-base, microsoft/deberta-base, etc.)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (16-32 recommended)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate (2e-5 to 5e-5 for transformers)')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max token length (128 tweets, 256 balance, 512 articles)')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Test set fraction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--true_csv', type=str, default='True.csv',
                        help='Path to True.csv')
    parser.add_argument('--fake_csv', type=str, default='Fake.csv',
                        help='Path to Fake.csv')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ROBERTA/DEBERTA FAKE NEWS DETECTOR - TRAINING")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}, Max tokens: {args.max_length}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # Load dataset
    texts, labels = load_isot_dataset(args.true_csv, args.fake_csv)
    
    # Split: 70% train, 15% val, 15% test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=(1 - 0.7), stratify=labels, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    print(f"\nData Split:")
    print(f"  Training: {len(train_texts)} ({len(train_texts)/len(texts)*100:.1f}%)")
    print(f"  Validation: {len(val_texts)} ({len(val_texts)/len(texts)*100:.1f}%)")
    print(f"  Test: {len(test_texts)} ({len(test_texts)/len(texts)*100:.1f}%)")
    
    # Initialize detector
    if 'deberta' in args.model.lower():
        print(f"\nInitializing DeBERTa detector...")
        detector = DeBertaFakeNewsDetector(model_name=args.model, device=args.device)
    else:
        print(f"\nInitializing RoBERTa detector...")
        detector = RobertaFakeNewsDetector(model_name=args.model, device=args.device)
    
    # Fine-tune
    best_val_f1 = detector.fine_tune(
        train_texts, train_labels,
        val_texts, val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
    
    # Test set evaluation
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    test_preds, test_confs = detector.batch_predict(test_texts, batch_size=args.batch_size)
    test_preds_int = [0 if p == 'FAKE' else 1 for p in test_preds]
    
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, roc_auc_score,
        confusion_matrix, classification_report
    )
    
    test_f1 = f1_score(test_labels, test_preds_int, average='macro')
    test_precision = precision_score(test_labels, test_preds_int, average='macro')
    test_recall = recall_score(test_labels, test_preds_int, average='macro')
    
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds_int).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    print(f"\n📊 Primary Metric")
    print(f"  F1 Score (macro): {test_f1:.4f}")
    
    print(f"\n📈 Secondary Metrics")
    print(f"  Precision (macro): {test_precision:.4f}")
    print(f"  Recall (macro): {test_recall:.4f}")
    print(f"  ROC-AUC: {roc_auc_score(test_labels, test_confs):.4f}")
    
    print(f"\n🎯 Operational Metrics")
    print(f"  False Positive Rate: {fpr:.2%} (how many fake we mark as real)")
    print(f"  False Negative Rate: {fnr:.2%} (how many real we mark as fake)")
    
    print(f"\n📋 Detailed Classification Report")
    print(classification_report(test_labels, test_preds_int, target_names=['FAKE', 'REAL']))
    
    print(f"\n✅ Acceptance Criteria")
    print(f"  F1 ≥ 0.95? {'✓ PASS' if test_f1 >= 0.95 else '✗ FAIL'}")
    print(f"  FPR ≤ 0.02? {'✓ PASS' if fpr <= 0.02 else '✗ FAIL'}")
    print(f"  FNR ≤ 0.02? {'✓ PASS' if fnr <= 0.02 else '✗ FAIL'}")
    
    # Save model
    model_name = 'deberta' if 'deberta' in args.model.lower() else 'roberta'
    save_path = f'models/{model_name}_best_f1_{test_f1:.4f}'
    detector.save_model(save_path)
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best model saved to: {save_path}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Ready for deployment!")
    print(f"{'='*70}\n")
    
    # Next steps
    print("📌 Next Steps:")
    print(f"   1. Compare with current ensemble (current best F1: 0.97)")
    print(f"   2. If F1 > 0.97, integrate into Streamlit app")
    print(f"   3. Add attention-based explainability (Phase 3)")
    print(f"   4. Deploy to production")


if __name__ == '__main__':
    main()
