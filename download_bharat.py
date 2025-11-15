"""
Download Bharat Fake News Dataset from Kaggle
==============================================
Combines with existing datasets for enhanced training
"""

import kagglehub
import pandas as pd
import os
from pathlib import Path
import sys
import io

# Fix Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*80)
print("BHARAT FAKE NEWS DATASET DOWNLOADER")
print("="*80 + "\n")

# Download
print("[1/3] Downloading dataset...", end=" ", flush=True)
try:
    path = kagglehub.dataset_download("man2191989/bharatfakenewskosh")
    print("OK\n")
    print(f"  Path: {path}\n")
except Exception as e:
    print(f"ERROR: {str(e)[:60]}\n")
    exit(1)

# Explore
print("[2/3] Exploring files...\n")
files = os.listdir(path)

for file in files:
    file_path = os.path.join(path, file)
    if os.path.isfile(file_path):
        size_mb = os.path.getsize(file_path) / (1024*1024)
        print(f"  - {file} ({size_mb:.1f} MB)")

# Process files (CSV or Excel)
print("\n[3/3] Processing data...\n")

data_files = [f for f in files if f.endswith(('.csv', '.xlsx', '.xls'))]
total_rows = 0

for data_file in data_files:
    file_path = os.path.join(path, data_file)
    print(f"  Loading {data_file}...", end=" ", flush=True)
    
    try:
        if data_file.endswith(('.xlsx', '.xls')):
            # Excel file - read all sheets
            xls = pd.ExcelFile(file_path)
            print(f"OK (Sheets: {xls.sheet_names})")
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                rows = len(df)
                cols = len(df.columns)
                total_rows += rows
                
                print(f"    Sheet '{sheet_name}': {rows} rows, {cols} cols")
                print(f"      Columns: {list(df.columns)[:5]}\n")
        else:
            df = pd.read_csv(file_path)
            rows = len(df)
            cols = len(df.columns)
            total_rows += rows
            
            print(f"OK ({rows} rows, {cols} cols)")
            print(f"    Columns: {list(df.columns)[:5]}\n")
        
    except Exception as e:
        print(f"ERROR: {str(e)[:40]}\n")

print("="*80)
print(f"TOTAL: {total_rows} rows across {len(data_files)} files")
print("="*80 + "\n")

print("Next: Integrate into training pipeline")
print("  Run: python train_with_bharat.py\n")
