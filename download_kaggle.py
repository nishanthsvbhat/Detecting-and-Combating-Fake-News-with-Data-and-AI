"""
Download and integrate Kaggle Fake News datasets
================================================
Combines Kaggle datasets with existing local datasets
"""

import kagglehub
import pandas as pd
import os
from pathlib import Path
import shutil
import sys

# Fix Unicode encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("KAGGLE DATASET DOWNLOADER")
print("="*80)

# Download Kaggle dataset
print("\n[1/3] Downloading Kaggle dataset...")
try:
    path = kagglehub.dataset_download("imbikramsaha/fake-real-news")
    print(f"[OK] Downloaded to: {path}")
except Exception as e:
    print(f"[ERROR] {e}")
    print("\nPlease configure Kaggle API:")
    print("  1. Visit: https://www.kaggle.com/settings/account")
    print("  2. Click 'Create New API Token'")
    print("  3. Place kaggle.json in ~/.kaggle/")
    print("  4. Run: kaggle configure")
    exit(1)

# List files in dataset
print("\n[2/3] Exploring dataset structure...")
dataset_files = os.listdir(path)
print(f"Files found: {len(dataset_files)}")
for file in dataset_files:
    file_path = os.path.join(path, file)
    if os.path.isfile(file_path):
        size_mb = os.path.getsize(file_path) / (1024*1024)
        print(f"  |- {file} ({size_mb:.1f} MB)")
    else:
        print(f"  |- {file}/ (directory)")

# Load and inspect CSV files
print("\n[3/3] Processing Kaggle data...")

kaggle_dfs = []

for file in dataset_files:
    file_path = os.path.join(path, file)
    
    if file.endswith('.csv'):
        print(f"  |- Loading {file}...", end=" ")
        try:
            df = pd.read_csv(file_path)
            print(f"[OK] ({len(df)} rows, {len(df.columns)} cols)")
            
            # Show columns
            print(f"      Columns: {list(df.columns)}")
            
            # Try to identify label and text columns
            has_label = any(col.lower() in ['label', 'class', 'target', 'category'] for col in df.columns)
            has_text = any(col.lower() in ['text', 'content', 'news', 'article', 'title', 'description'] for col in df.columns)
            
            if has_label and has_text:
                kaggle_dfs.append({
                    'filename': file,
                    'dataframe': df,
                    'rows': len(df),
                    'columns': list(df.columns)
                })
        except Exception as e:
            print(f"[ERROR] {e}")

print(f"\n[OK] Successfully loaded {len(kaggle_dfs)} datasets from Kaggle")

# Save summary
print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)

for i, dataset in enumerate(kaggle_dfs, 1):
    print(f"\nDataset {i}: {dataset['filename']}")
    print(f"  . Rows: {dataset['rows']}")
    print(f"  . Columns: {', '.join(dataset['columns'][:5])}{'...' if len(dataset['columns']) > 5 else ''}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Update train_production.py to include Kaggle data
2. Combine with existing datasets
3. Retrain models with combined data
4. Test the improved system

Run: python train_production_with_kaggle.py
""")
