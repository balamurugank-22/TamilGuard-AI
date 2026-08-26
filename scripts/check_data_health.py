import pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

df = pd.read_csv("data/processed/weak_bio_labels.csv")
sents = df.drop_duplicates("sent_id")

print("=== TRAINING DATA HEALTH CHECK ===")
print(f"Total sentences  : {sents.shape[0]:,}")
print(f"Labeled (matched): {(sents.needs_review=='False').sum():,}  ({(sents.needs_review=='False').mean():.1%})")
print(f"Unlabeled (no match): {(sents.needs_review=='True').sum():,}  ({(sents.needs_review=='True').mean():.1%})")

print("\n=== TAG DISTRIBUTION ===")
print(df['tag'].value_counts())

print("\n=== MATCH TYPE BREAKDOWN ===")
print(df[df.tag!='O']['match_type'].value_counts())

print("\n=== CATEGORY BREAKDOWN ===")
print(df[df.tag!='O']['category'].value_counts())

print("\n=== AVG TOKENS PER SENTENCE ===")
print(f"Mean token length: {df.groupby('sent_id').size().mean():.1f}")
print(f"Max  token length: {df.groupby('sent_id').size().max()}")
