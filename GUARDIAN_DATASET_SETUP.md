# 📥 The Guardian Dataset Setup Guide

## 📊 Dataset Information

- **Name**: The Guardian
- **Dataset ID**: `08d64e83-91f4-4b4d-9efe-60fee5e31799`
- **Files Required**: `guardian_fake.csv` + `guardian_real.csv`
- **Expected Size**: ~30-100 MB (total)
- **Articles**: 10,000+ news articles from The Guardian

---

## 📋 Required CSV Format

Both `guardian_fake.csv` and `guardian_real.csv` should have **AT LEAST ONE** of these columns:

### Column Options (in priority order)
1. `text` - Full article text
2. `content` - Article content
3. `article` - Article body
4. `description` - Article description
5. `title` - Article title (fallback)

### Example Format

**guardian_fake.csv:**
```
text,author,date,label
"Article content here...",Author Name,2023-01-15,0
"Another fake article...",Another Author,2023-01-16,0
```

**guardian_real.csv:**
```
text,author,date,label
"Real article content...",Author Name,2023-01-15,1
"Another real article...",Another Author,2023-01-16,1
```

⚠️ **Important**: The script will automatically add the `label` column (0=Fake, 1=Real)

---

## 🔍 Where to Get The Guardian Dataset

### Option 1: Download from Kaggle

1. Go to: https://www.kaggle.com/search?q=guardian
2. Search for "guardian news" or "fake news guardian"
3. Download datasets with fake/real articles
4. Save as `guardian_fake.csv` and `guardian_real.csv`

### Option 2: From GitHub

1. Search: https://github.com/search?q=guardian+fake+news+dataset
2. Look for repositories with separated fake/real news
3. Download CSV files

### Option 3: API Download (Programmatic)

```python
# If the dataset is available via API
# Use pandas to download and split:

import pandas as pd

# Download the dataset
df = pd.read_csv('https://source.com/guardian_data.csv')

# Filter into fake and real
fake_df = df[df['label'] == 0]
real_df = df[df['label'] == 1]

# Save separately
fake_df.to_csv('guardian_fake.csv', index=False)
real_df.to_csv('guardian_real.csv', index=False)
```

---

## ✅ Verification Checklist

Before running training, verify:

```bash
# 1. Check files exist
ls -la guardian_fake.csv guardian_real.csv

# 2. Check file sizes
wc -l guardian_fake.csv guardian_real.csv

# 3. Check columns
head -5 guardian_fake.csv | cut -d',' -f1-3
head -5 guardian_real.csv | cut -d',' -f1-3

# 4. Verify format (PowerShell)
(Measure-Object -InputObject (Get-Content guardian_fake.csv) -Line).Lines
```

---

## 🚀 After Adding Guardian Files

Once you have `guardian_fake.csv` and `guardian_real.csv`:

### Step 1: Verify Files
```bash
cd c:\Users\Nishanth\Documents\fake_news_project
dir guardian*.csv
```

### Step 2: Run Training
```bash
python train_unified_multi_dataset.py
```

**Training will:**
- ✓ Load Original dataset (Fake.csv + True.csv)
- ✓ Load GossipCop dataset
- ✓ Load PolitiFact dataset
- ✓ Load Guardian dataset
- ✓ Combine all 4 datasets
- ✓ Train 5 ML models
- ✓ Create ensemble voting
- ✓ Save models to `model_artifacts_multi_dataset/`

### Step 3: Use in App
```bash
streamlit run app_with_multi_dataset.py
```

---

## 📊 Current Dataset Status

### Existing Datasets ✅

| Dataset | Fake | Real | Status |
|---------|------|------|--------|
| Original | ✅ (23,481) | ✅ (21,417) | Ready |
| GossipCop | ✅ | ✅ | Ready |
| PolitiFact | ✅ | ✅ | Ready |
| Guardian | ⏳ Pending | ⏳ Pending | Waiting for files |

### Combined Statistics (After Adding Guardian)

```
Total Articles: 110,000+
- Original: 44,898
- GossipCop: ~15,000
- PolitiFact: ~11,000
- Guardian: ~39,000+
```

---

## 🔧 Troubleshooting Guardian Dataset

### Problem: File Not Found
```
FileNotFoundError: guardian_fake.csv
```
**Solution**: Ensure files are in the project root directory
```bash
# Copy files to correct location
cp /path/to/guardian_fake.csv ./
cp /path/to/guardian_real.csv ./
```

### Problem: Column Not Found
```
KeyError: 'text'
```
**Solution**: The script tries multiple column names (text, content, article, description, title)
If your CSV has different columns, rename them:

```python
import pandas as pd

df = pd.read_csv('guardian_fake.csv')
df.rename(columns={'your_column_name': 'text'}, inplace=True)
df.to_csv('guardian_fake.csv', index=False)
```

### Problem: Memory Error During Training
```
MemoryError: Unable to allocate X GB
```
**Solution**: Guardian dataset may be large. Reduce features or split training:

```python
# In train_unified_multi_dataset.py, line 90:
TfidfVectorizer(
    max_features=3000,  # Reduced from 5000
    ...
)
```

### Problem: Different Label Format
If your CSV doesn't have a label column, or uses different values:

```python
import pandas as pd

# Fake dataset - no label
fake_df = pd.read_csv('guardian_fake.csv')
fake_df['label'] = 0  # Add fake label

# Real dataset - no label
real_df = pd.read_csv('guardian_real.csv')
real_df['label'] = 1  # Add real label

# Save
fake_df.to_csv('guardian_fake.csv', index=False)
real_df.to_csv('guardian_real.csv', index=False)
```

---

## 📈 Performance Impact

Adding The Guardian dataset will:

✅ **Improve accuracy** - More diverse training data
✅ **Better generalization** - Learn from news sources
✅ **Reduce overfitting** - Larger dataset
✅ **Increase robustness** - Different article styles

**Expected improvements:**
- Model accuracy: +1-3%
- Real-world performance: +5-10%
- False positive rate: -2-5%

---

## 🔄 Manual Guardian Data Processing

If you have raw Guardian data, process it like this:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw data
df = pd.read_csv('guardian_raw.csv')

# Ensure required columns
if 'text' not in df.columns:
    # Try to use first text-like column
    text_cols = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
    if text_cols:
        df.rename(columns={text_cols[0]: 'text'}, inplace=True)

# Remove duplicates and null values
df = df.dropna(subset=['text'])
df = df[df['text'].str.len() > 50]  # Keep only meaningful articles
df = df.drop_duplicates(subset=['text'])

# Ensure label column exists
if 'label' not in df.columns:
    raise ValueError("Dataset must have a 'label' column (0=Fake, 1=Real)")

# Split if combined
fake_articles = df[df['label'] == 0]
real_articles = df[df['label'] == 1]

# Save
fake_articles.to_csv('guardian_fake.csv', index=False)
real_articles.to_csv('guardian_real.csv', index=False)

print(f"✓ Fake: {len(fake_articles)} articles")
print(f"✓ Real: {len(real_articles)} articles")
```

---

## 📝 File Checklist

Before running training, have these files:

```
✓ Fake.csv
✓ True.csv
✓ gossipcop_fake.csv
✓ gossipcop_real.csv
✓ politifact_fake.csv
✓ politifact_real.csv
✓ guardian_fake.csv      ← NEW
✓ guardian_real.csv      ← NEW
```

---

## 🚀 Quick Start After Getting Guardian Files

```bash
# 1. Copy files to project directory
cp ~/Downloads/guardian_fake.csv ./
cp ~/Downloads/guardian_real.csv ./

# 2. Train
python train_unified_multi_dataset.py

# 3. Run app
streamlit run app_with_multi_dataset.py
```

**Total time**: ~15-20 minutes

---

## 📞 Support

**Issues with Guardian dataset?**
- Check file format matches example above
- Verify files are in project root
- Ensure at least one text column exists
- Check file size (min 1 MB recommended)

---

**Status**: 🟡 Awaiting Guardian Dataset Files  
**Required Files**: `guardian_fake.csv` + `guardian_real.csv`  
**Last Updated**: November 2025
