"""Quick script to check what labels are in the processed data."""

import pandas as pd
import ast
from collections import Counter

# Load the data
df = pd.read_csv('processed_data/train.csv')

print("=" * 70)
print("LABEL ANALYSIS")
print("=" * 70)

print(f"\nTotal rows: {len(df)}")
print(f"Rows with labels: {df['labels'].notna().sum()}")
print(f"Rows without labels: {df['labels'].isna().sum()}")

# Extract numeric IDs
print("\n" + "-" * 70)
print("Extracting numeric label IDs...")
print("-" * 70)

all_ids = []
label_strings = df['labels'].dropna().head(1000)  # Check first 1000 rows

for label_str in label_strings:
    try:
        parsed = ast.literal_eval(str(label_str))
        if isinstance(parsed, list):
            all_ids.extend(parsed)
        else:
            all_ids.append(parsed)
    except:
        pass

if all_ids:
    id_counts = Counter(all_ids)
    unique_ids = sorted(id_counts.keys())
    
    print(f"\nFound {len(unique_ids)} unique label IDs")
    print(f"Range: {min(unique_ids)} to {max(unique_ids)}")
    print(f"\nTop 20 most common label IDs:")
    for label_id, count in id_counts.most_common(20):
        print(f"  ID {label_id}: {count} occurrences")
    
    print(f"\nAll unique IDs: {unique_ids}")
    
    # Check if this matches GoEmotions (0-27)
    if set(unique_ids) <= set(range(28)):
        print("\n" + "=" * 70)
        print("✓ This matches GoEmotions dataset (IDs 0-27)")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠ Warning: IDs don't match GoEmotions exactly")
        print("=" * 70)
else:
    print("\n⚠ Could not extract numeric IDs. Labels might already be converted.")

print("\n" + "-" * 70)
print("Sample labels from CSV:")
print("-" * 70)
for i, label in enumerate(df['labels'].dropna().head(10), 1):
    print(f"  {i}: {repr(label)}")

