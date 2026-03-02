"""Quick script to verify processed data."""
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent / "processed_data"

print("=" * 70)
print("Data Verification")
print("=" * 70)

for split in ['train', 'validation', 'test']:
    filepath = data_dir / f"{split}.csv"
    if not filepath.exists():
        print(f"\n[WARN] {split}.csv not found")
        continue
    
    df = pd.read_csv(filepath)
    crisis_count = (df['label'] == 1).sum()
    non_crisis_count = (df['label'] == 0).sum()
    
    print(f"\n{split.upper()}:")
    print(f"  Total: {len(df)}")
    print(f"  Crisis (1): {crisis_count}")
    print(f"  Non-crisis (0): {non_crisis_count}")
    print(f"  Balance: {crisis_count/(crisis_count+non_crisis_count)*100:.1f}% / {non_crisis_count/(crisis_count+non_crisis_count)*100:.1f}%")
    
    if split == 'train':
        print(f"\n  Sample crisis text:")
        sample = df[df['label']==1]['text'].iloc[0]
        print(f"    {sample[:150]}...")
        print(f"\n  Sample non-crisis text:")
        sample = df[df['label']==0]['text'].iloc[0]
        print(f"    {sample[:150]}...")

print("\n" + "=" * 70)
print("Verification complete!")
print("=" * 70)

