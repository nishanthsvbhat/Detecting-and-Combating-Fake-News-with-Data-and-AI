import pandas as pd

print("=" * 60)
print("CHECKING RSS NEWS DATA")
print("=" * 60)

df = pd.read_csv('rss_news.csv')

print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst row:")
for col in df.columns:
    print(f"  {col}: {str(df.iloc[0][col])[:80]}...")
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nValue counts (if label column exists):")
if 'label' in df.columns:
    print(df['label'].value_counts())
elif 'class' in df.columns:
    print(df['class'].value_counts())
