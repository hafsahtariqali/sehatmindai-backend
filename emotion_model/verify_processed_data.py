"""Verify the processed data has correct labels."""

import pandas as pd
from collections import Counter

print("=" * 70)
print("VERIFYING PROCESSED DATA")
print("=" * 70)

# Load data
train_df = pd.read_csv('processed_data/train.csv')
val_df = pd.read_csv('processed_data/validation.csv')
test_df = pd.read_csv('processed_data/test.csv')

print(f"\n[FILE SIZES]")
print(f"Train: {len(train_df):,} rows")
print(f"Validation: {len(val_df):,} rows")
print(f"Test: {len(test_df):,} rows")
print(f"Total: {len(train_df) + len(val_df) + len(test_df):,} rows")

print(f"\n[LABEL COVERAGE]")
print(f"Train - Rows with labels: {train_df['labels'].notna().sum():,} / {len(train_df):,}")
print(f"Validation - Rows with labels: {val_df['labels'].notna().sum():,} / {len(val_df):,}")
print(f"Test - Rows with labels: {test_df['labels'].notna().sum():,} / {len(test_df):,}")

print(f"\n[EMOTION DISTRIBUTION - TRAIN SET]")
all_labels = []
for label in train_df['labels'].dropna():
    if pd.notna(label):
        all_labels.extend(str(label).split('|'))

label_counts = Counter(all_labels)
total_label_occurrences = sum(label_counts.values())

print(f"Total emotion occurrences: {total_label_occurrences:,}")
print(f"\nEmotion distribution:")
for emotion in ['joy', 'sadness', 'anger', 'fear', 'anxiety', 'loneliness', 'neutral']:
    count = label_counts.get(emotion, 0)
    pct = count / total_label_occurrences * 100 if total_label_occurrences > 0 else 0
    print(f"  {emotion:12s}: {count:6,} ({pct:5.1f}%)")

print(f"\n[MULTI-LABEL STATISTICS]")
train_multi = train_df['labels'].dropna().apply(lambda x: '|' in str(x)).sum()
train_single = train_df['labels'].dropna().apply(lambda x: '|' not in str(x)).sum()
print(f"Single-label examples: {train_single:,} ({train_single/len(train_df)*100:.1f}%)")
print(f"Multi-label examples: {train_multi:,} ({train_multi/len(train_df)*100:.1f}%)")

print(f"\n[SAMPLE LABELS - TRAIN]")
sample_labels = train_df['labels'].dropna().head(30).tolist()
for i, label in enumerate(sample_labels, 1):
    print(f"  {i:2d}. {label}")

print(f"\n[VALIDATION CHECK]")
val_sample = val_df['labels'].dropna().head(10).tolist()
print("Sample validation labels:")
for label in val_sample:
    print(f"  - {label}")

print(f"\n[TEST CHECK]")
test_sample = test_df['labels'].dropna().head(10).tolist()
print("Sample test labels:")
for label in test_sample:
    print(f"  - {label}")

# Check for any invalid labels
print(f"\n[VALIDATION - CHECKING FOR ISSUES]")
valid_emotions = {'joy', 'sadness', 'anger', 'fear', 'anxiety', 'loneliness', 'neutral'}
all_unique_labels = set()
for label in train_df['labels'].dropna():
    if pd.notna(label):
        emotions = str(label).split('|')
        all_unique_labels.update([e.strip().lower() for e in emotions])

invalid_labels = all_unique_labels - valid_emotions
if invalid_labels:
    print(f"  [WARN] Found invalid emotion labels: {invalid_labels}")
else:
    print(f"  [OK] All labels are valid emotions!")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

