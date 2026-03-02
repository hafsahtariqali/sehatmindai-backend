"""
Preprocessing script for crisis detection dataset.

This script:
- Loads GoEmotions dataset and identifies crisis-related samples (extreme sadness, despair, self-harm)
- Loads CounselChat messages as non-crisis samples
- Loads custom crisis phrases for additional crisis examples
- Balances the dataset to avoid class imbalance
- Saves processed data with columns: text, label
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import Counter
import random


# Define paths
BASE_DIR = Path(__file__).parent.parent  # chatbot_backend/
DATASETS_DIR = BASE_DIR / "datasets"
LABEL_MAP_PATH = Path(__file__).parent / "label_map.json"
OUTPUT_DIR = Path(__file__).parent / "processed_data"
CUSTOM_CRISIS_DIR = DATASETS_DIR / "custom_crisis"

# GoEmotions label IDs that indicate crisis
# Based on GoEmotions: labels 9=disappointment, 16=grief, 24=remorse, 25=sadness
# We'll also look for self-harm keywords in the text
CRISIS_GOEMOTIONS_LABELS = [9, 16, 24, 25]  # disappointment, grief, remorse, sadness

# Keywords that indicate crisis/self-harm
# IMPORTANT: Use specific phrases to avoid false positives
# e.g., "cutting" alone matches "cutting people" vs "cutting myself"
CRISIS_KEYWORDS = [
    # Direct self-harm
    'suicide', 'suicidal', 'kill myself', 'end my life', 'want to die', 'killing myself',
    'hurt myself', 'self harm', 'self-harm', 'cut myself', 'cutting myself', 
    'overdose', 'overdosing', 'end it all', 'better off dead',
    # Mental health crisis indicators
    'nothing matters', 'no point', 'hopeless', 'despair', 'no hope',
    'dont want to live', "don't want to live", 'cant go on', "can't go on"
]


def load_label_map(label_map_path: Path) -> Dict:
    """Load the label mapping from JSON file."""
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    return label_map


def find_dataset_file(dataset_name: str, split: str) -> Optional[Path]:
    """Find dataset file for given dataset and split."""
    dataset_dir = DATASETS_DIR / dataset_name
    
    if not dataset_dir.exists():
        return None
    
    # Try different naming patterns
    patterns = [
        f"{dataset_name}_{split}.csv",
        f"{split}.csv",
        f"{split}.tsv",
        f"train.csv" if split == "train" else None,
        f"validation.csv" if split == "validation" else None,
        f"valid.csv" if split == "validation" else None,
        f"test.csv" if split == "test" else None,
    ]
    
    for pattern in patterns:
        if pattern is None:
            continue
        filepath = dataset_dir / pattern
        if filepath.exists():
            return filepath
    
    # Look for any CSV file in the directory
    csv_files = list(dataset_dir.glob("*.csv"))
    if csv_files and split == "train":
        return csv_files[0]
    
    return None


def contains_crisis_keywords(text: str) -> bool:
    """
    Check if text contains crisis-related keywords.
    Uses specific phrases to avoid false positives.
    """
    if pd.isna(text):
        return False
    
    text_lower = str(text).lower()
    
    # Check for crisis keywords
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return True
    
    # Check for patterns that indicate mental health crisis
    # (not just general sadness, but urgent/acute distress)
    crisis_patterns = [
        r'\bdepressed\b.*\b(dont|don\'t|can\'t|cant)\s*(eat|sleep|function|cope)',
        r'\bsuicidal\s+thoughts?\b',
        r'\b(thinking|thoughts?)\s+about\s+(suicide|dying|ending)',
        r'\b(dont|don\'t)\s*(want|wanna)\s+to\s+live',
        r'\bno\s+reason\s+to\s+live',
        r'\b(cant|can\'t)\s+(take|handle|do)\s+(it|this)\s+anymore'
    ]
    
    import re
    for pattern in crisis_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def parse_labels(label_str: str) -> List[int]:
    """Parse label string from GoEmotions format."""
    if pd.isna(label_str):
        return []
    
    try:
        import ast
        parsed = ast.literal_eval(str(label_str))
        if isinstance(parsed, list):
            return [int(x) for x in parsed if isinstance(x, (int, float))]
        elif isinstance(parsed, (int, float)):
            return [int(parsed)]
    except:
        pass
    
    return []


def load_goemotions_crisis_samples(split: str) -> pd.DataFrame:
    """Load crisis samples from GoEmotions dataset."""
    filepath = find_dataset_file("goemotions", split)
    
    if filepath is None:
        print(f"  [WARN] GoEmotions {split} file not found")
        return pd.DataFrame(columns=['text', 'label'])
    
    print(f"  Loading GoEmotions {split}...")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    except:
        try:
            df = pd.read_csv(filepath, sep='\t', encoding='utf-8', on_bad_lines='skip')
        except:
            df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip')
    
    if df.empty:
        print(f"    Empty file: {filepath.name}")
        return pd.DataFrame(columns=['text', 'label'])
    
    print(f"    Loaded {len(df)} rows")
    
    # Normalize columns
    text_col = None
    label_col = None
    
    for col in ['text', 'sentence', 'comment', 'post']:
        if col in df.columns:
            text_col = col
            break
    
    for col in ['labels', 'label', 'emotion', 'emotion_id']:
        if col in df.columns:
            label_col = col
            break
    
    if text_col is None or label_col is None:
        print(f"    [WARN] Could not find text or label columns in {filepath.name}")
        return pd.DataFrame(columns=['text', 'label'])
    
    df = df.rename(columns={text_col: 'text', label_col: 'labels'})
    
    # Filter for crisis samples
    # IMPORTANT: Only use STRICT keyword matching for crisis detection
    # GoEmotions emotion labels don't indicate if the TEXT itself is a crisis statement
    # For example: "I'm sorry that happened" might have sadness label but isn't a crisis
    crisis_samples = []
    
    for idx, row in df.iterrows():
        text = str(row['text']).strip().lower()
        labels = parse_labels(row['labels'])
        
        # ONLY use keyword-based detection - don't rely on emotion labels alone
        # Emotion labels indicate emotions expressed, not whether it's a crisis statement
        has_crisis_keywords = contains_crisis_keywords(text)
        
        # Only mark as crisis if it contains explicit crisis keywords
        # This filters out generic empathetic responses that happen to have sadness labels
        if has_crisis_keywords:
            crisis_samples.append({
                'text': str(row['text']).strip(),  # Keep original case
                'label': 1  # crisis
            })
    
    crisis_df = pd.DataFrame(crisis_samples)
    print(f"    Found {len(crisis_df)} crisis samples (keyword-filtered only)")
    
    return crisis_df


def load_counselchat_samples_split(split: str, max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CounselChat samples and split into crisis and non-crisis.
    
    Returns:
        Tuple of (crisis_df, non_crisis_df)
    """
    filepath = find_dataset_file("counsel_chat", split)
    
    if filepath is None:
        print(f"  [WARN] CounselChat {split} file not found")
        return pd.DataFrame(columns=['text', 'label']), pd.DataFrame(columns=['text', 'label'])
    
    print(f"  Loading CounselChat {split}...")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    except:
        try:
            df = pd.read_csv(filepath, sep='\t', encoding='utf-8', on_bad_lines='skip')
        except:
            df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip')
    
    if df.empty:
        print(f"    Empty file: {filepath.name}")
        return pd.DataFrame(columns=['text', 'label']), pd.DataFrame(columns=['text', 'label'])
    
    print(f"    Loaded {len(df)} rows")
    
    # Normalize columns
    text_col = None
    for col in ['text', 'message', 'question', 'input', 'utterance']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        # Use first column as text
        text_col = df.columns[0]
    
    df = df.rename(columns={text_col: 'text'})
    
    # Split into crisis and non-crisis
    crisis_samples = []
    non_crisis_samples = []
    
    import re
    
    for idx, row in df.iterrows():
        text = str(row['text']).strip()
        text_lower = text.lower()
        
        # Check for crisis keywords first
        has_crisis_keywords = contains_crisis_keywords(text_lower)
        
        # Check for mental health distress patterns
        mental_health_distress_patterns = [
            r'\bdepressed\b',  # mentions depression
            r'\b(dont|don\'t)\s+(sleep|eat)',  # can't sleep/eat
            r'\b(no|not)\s+energy\s+to',  # no energy
            r'\b(not|cant|can\'t)\s+(sleep|eat|function)',  # can't function
            r'\bsuicidal\b',  # suicidal thoughts
            r'\boverwhelmed\b.*\b(dont|can\'t)',  # overwhelmed and can't cope
            r'\b(cant|can\'t)\s+cope',  # can't cope
            r'\bmental\s+breakdown\b',  # mental breakdown
            r'\b(self\s*)?harm\b',  # self-harm
            r'\bthinking\s+about\s+(dying|suicide)',  # thinking about suicide
        ]
        
        has_distress_pattern = any(re.search(pattern, text_lower) for pattern in mental_health_distress_patterns)
        
        # Check for urgent/problem language
        urgent_keywords = [
            'abuse', 'verbal abuse', 'physical abuse', 'hurting me',
            'crying every day', 'every day i cry', 'cry every night',
            'not sleeping', 'not eating', 'dont eat', 'dont sleep'
        ]
        
        has_urgent_language = any(keyword in text_lower for keyword in urgent_keywords)
        
        # Check if it's personal mental health distress (not general question)
        is_personal_distress = False
        if ' i ' in text_lower or text_lower.startswith('i '):
            personal_distress_words = ['sad', 'anxious', 'worried', 'stressed', 'lonely', 
                                     'empty', 'numb', 'feel', 'feeling']
            if any(word in text_lower for word in personal_distress_words):
                is_personal_distress = True
        
        # Mark as crisis if it has crisis indicators
        if has_crisis_keywords or has_distress_pattern or has_urgent_language or is_personal_distress:
            crisis_samples.append({
                'text': text,
                'label': 1  # crisis
            })
        else:
            # Check if it's a general question (about others or general advice)
            general_question_indicators = [
                'my partner', 'my friend', 'my spouse', 'my boyfriend', 'my girlfriend',
                'how do i help', 'how can i help', 'what should i do about',
                'relationship advice', 'communication', 'boundaries'
            ]
            
            is_general_question = any(indicator in text_lower for indicator in general_question_indicators)
            
            # Only include as non-crisis if it's a general question
            if is_general_question:
                non_crisis_samples.append({
                    'text': text,
                    'label': 0  # not_crisis
                })
    
    crisis_df = pd.DataFrame(crisis_samples)
    non_crisis_df = pd.DataFrame(non_crisis_samples)
    
    # Limit samples if specified (applies to each separately)
    if max_samples:
        if len(crisis_df) > max_samples:
            crisis_df = crisis_df.sample(n=max_samples, random_state=42)
        if len(non_crisis_df) > max_samples:
            non_crisis_df = non_crisis_df.sample(n=max_samples, random_state=42)
    
    print(f"    Found {len(crisis_df)} crisis samples")
    print(f"    Found {len(non_crisis_df)} non-crisis samples")
    
    return crisis_df, non_crisis_df


def load_suicide_ideation_dataset() -> pd.DataFrame:
    """
    Load suicide_ideation dataset.
    Labels: 'suicide' → crisis (1), 'non-suicide' → non-crisis (0)
    """
    filepath = DATASETS_DIR / "suicide_ideation" / "Suicide_Detection.csv"
    
    if not filepath.exists():
        print(f"  [WARN] Suicide Ideation dataset not found: {filepath}")
        return pd.DataFrame(columns=['text', 'label'])
    
    print(f"  Loading Suicide Ideation dataset...")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    except:
        try:
            df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip')
        except Exception as e:
            print(f"    [ERROR] Failed to load: {e}")
            return pd.DataFrame(columns=['text', 'label'])
    
    if df.empty:
        print(f"    [WARN] Empty file")
        return pd.DataFrame(columns=['text', 'label'])
    
    print(f"    Loaded {len(df)} rows")
    
    # Normalize columns
    if 'text' not in df.columns:
        if 'statement' in df.columns:
            df = df.rename(columns={'statement': 'text'})
        elif 'message' in df.columns:
            df = df.rename(columns={'message': 'text'})
        else:
            print(f"    [WARN] Could not find text column. Available columns: {df.columns.tolist()}")
            return pd.DataFrame(columns=['text', 'label'])
    
    if 'class' not in df.columns:
        print(f"    [WARN] Could not find label column. Available columns: {df.columns.tolist()}")
        return pd.DataFrame(columns=['text', 'label'])
    
    # Normalize labels: 'suicide' → 1, 'non-suicide' → 0
    df['label'] = df['class'].apply(lambda x: 1 if str(x).lower().strip() == 'suicide' else 0)
    
    # Keep only text and label columns
    result_df = df[['text', 'label']].copy()
    
    # Clean text
    result_df['text'] = result_df['text'].astype(str).str.strip()
    result_df = result_df[result_df['text'] != '']
    
    # Remove duplicates
    result_df = result_df.drop_duplicates(subset=['text'], keep='first')
    
    print(f"    Processed {len(result_df)} samples")
    print(f"      Crisis (suicide): {(result_df['label'] == 1).sum()}")
    print(f"      Non-crisis (non-suicide): {(result_df['label'] == 0).sum()}")
    
    return result_df


def load_sentiment_analysis_dataset() -> pd.DataFrame:
    """
    Load sentiment_analysis dataset.
    Labels: 'Suicidal' → crisis (1), everything else → non-crisis (0)
    """
    filepath = DATASETS_DIR / "sentiment_analysis" / "Combined Data.csv"
    
    if not filepath.exists():
        print(f"  [WARN] Sentiment Analysis dataset not found: {filepath}")
        return pd.DataFrame(columns=['text', 'label'])
    
    print(f"  Loading Sentiment Analysis dataset...")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    except:
        try:
            df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip')
        except Exception as e:
            print(f"    [ERROR] Failed to load: {e}")
            return pd.DataFrame(columns=['text', 'label'])
    
    if df.empty:
        print(f"    [WARN] Empty file")
        return pd.DataFrame(columns=['text', 'label'])
    
    print(f"    Loaded {len(df)} rows")
    
    # Normalize columns
    if 'statement' not in df.columns:
        if 'text' in df.columns:
            df = df.rename(columns={'text': 'statement'})
        elif 'message' in df.columns:
            df = df.rename(columns={'message': 'statement'})
        else:
            print(f"    [WARN] Could not find text column. Available columns: {df.columns.tolist()}")
            return pd.DataFrame(columns=['text', 'label'])
    
    df = df.rename(columns={'statement': 'text'})
    
    if 'status' not in df.columns:
        print(f"    [WARN] Could not find label column. Available columns: {df.columns.tolist()}")
        return pd.DataFrame(columns=['text', 'label'])
    
    # Normalize labels: 'Suicidal' → 1, everything else → 0
    df['label'] = df['status'].apply(lambda x: 1 if str(x).lower().strip() == 'suicidal' else 0)
    
    # Keep only text and label columns
    result_df = df[['text', 'label']].copy()
    
    # Clean text
    result_df['text'] = result_df['text'].astype(str).str.strip()
    result_df = result_df[result_df['text'] != '']
    
    # Remove duplicates
    result_df = result_df.drop_duplicates(subset=['text'], keep='first')
    
    print(f"    Processed {len(result_df)} samples")
    print(f"      Crisis (Suicidal): {(result_df['label'] == 1).sum()}")
    print(f"      Non-crisis (other): {(result_df['label'] == 0).sum()}")
    
    return result_df


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df: DataFrame to split
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
        test_ratio: Proportion for test (default 0.1)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle with fixed seed for reproducibility
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_df = df_shuffled[:n_train].copy()
    val_df = df_shuffled[n_train:n_train + n_val].copy()
    test_df = df_shuffled[n_train + n_val:].copy()
    
    return train_df, val_df, test_df


def load_custom_crisis_phrases() -> pd.DataFrame:
    """Load custom crisis phrases from custom_crisis directory."""
    if not CUSTOM_CRISIS_DIR.exists():
        print(f"  [WARN] Custom crisis directory not found: {CUSTOM_CRISIS_DIR}")
        return pd.DataFrame(columns=['text', 'label'])
    
    print(f"  Loading custom crisis phrases...")
    
    crisis_samples = []
    
    # Look for CSV or text files
    csv_files = list(CUSTOM_CRISIS_DIR.glob("*.csv"))
    txt_files = list(CUSTOM_CRISIS_DIR.glob("*.txt"))
    
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
            
            # Find text column
            text_col = None
            for col in ['text', 'phrase', 'sentence', 'message']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                text_col = df.columns[0]
            
            for text in df[text_col].dropna():
                crisis_samples.append({
                    'text': str(text).strip(),
                    'label': 1  # crisis
                })
        except Exception as e:
            print(f"    [ERROR] Error loading {filepath.name}: {e}")
    
    for filepath in txt_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        crisis_samples.append({
                            'text': line,
                            'label': 1  # crisis
                        })
        except Exception as e:
            print(f"    [ERROR] Error loading {filepath.name}: {e}")
    
    custom_df = pd.DataFrame(crisis_samples)
    print(f"    Loaded {len(custom_df)} custom crisis phrases")
    
    return custom_df


def balance_dataset(df: pd.DataFrame, target_ratio: float = 0.5, random_state: int = 42) -> pd.DataFrame:
    """
    Balance dataset by adjusting class distribution.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        target_ratio: Target ratio of crisis samples (0.5 = balanced)
        random_state: Random seed for reproducibility
    
    Returns:
        Balanced DataFrame
    """
    crisis_df = df[df['label'] == 1].copy()
    non_crisis_df = df[df['label'] == 0].copy()
    
    n_crisis = len(crisis_df)
    n_non_crisis = len(non_crisis_df)
    
    print(f"\n  Original distribution:")
    print(f"    Crisis: {n_crisis}")
    print(f"    Non-crisis: {n_non_crisis}")
    print(f"    Ratio: {n_crisis / (n_crisis + n_non_crisis):.2%}")
    
    if n_crisis == 0 or n_non_crisis == 0:
        print(f"  [WARN] Cannot balance: one class is empty")
        return df
    
    # Calculate target sizes
    if n_crisis < n_non_crisis:
        # Crisis is minority - keep all, subsample non-crisis
        target_non_crisis = int(n_crisis / target_ratio * (1 - target_ratio))
        target_non_crisis = min(target_non_crisis, n_non_crisis)
        
        non_crisis_df = non_crisis_df.sample(n=target_non_crisis, random_state=random_state)
    else:
        # Non-crisis is minority - keep all, subsample crisis
        target_crisis = int(n_non_crisis / (1 - target_ratio) * target_ratio)
        target_crisis = min(target_crisis, n_crisis)
        
        crisis_df = crisis_df.sample(n=target_crisis, random_state=random_state)
    
    balanced_df = pd.concat([crisis_df, non_crisis_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n_crisis_balanced = (balanced_df['label'] == 1).sum()
    n_non_crisis_balanced = (balanced_df['label'] == 0).sum()
    
    print(f"\n  Balanced distribution:")
    print(f"    Crisis: {n_crisis_balanced}")
    print(f"    Non-crisis: {n_non_crisis_balanced}")
    print(f"    Ratio: {n_crisis_balanced / len(balanced_df):.2%}")
    
    return balanced_df


def main():
    """Main preprocessing function."""
    print("=" * 70)
    print("Crisis Detection Dataset Preprocessing")
    print("=" * 70)
    
    # Load label map
    print("\n[1] Loading label mapping...")
    label_map = load_label_map(LABEL_MAP_PATH)
    print(f"  [OK] Loaded label map")
    print(f"    Labels: {list(label_map['label_to_id'].keys())}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[2] Output directory: {OUTPUT_DIR}")
    
    print("\n[3] Loading datasets...")
    
    all_datasets = []
    
    # 1. Load Suicide Ideation dataset
    suicide_ideation_df = load_suicide_ideation_dataset()
    if not suicide_ideation_df.empty:
        all_datasets.append(suicide_ideation_df)
    
    # 2. Load Sentiment Analysis dataset
    sentiment_df = load_sentiment_analysis_dataset()
    if not sentiment_df.empty:
        all_datasets.append(sentiment_df)
    
    # 3. Load GoEmotions crisis samples (optional, for additional data)
    goemotions_df = load_goemotions_crisis_samples('train')
    if not goemotions_df.empty:
        all_datasets.append(goemotions_df)
    
    # 4. Load CounselChat samples (optional, for additional data)
    counselchat_crisis_df, counselchat_non_crisis_df = load_counselchat_samples_split('train', max_samples=5000)
    if not counselchat_crisis_df.empty:
        all_datasets.append(counselchat_crisis_df)
    if not counselchat_non_crisis_df.empty:
        all_datasets.append(counselchat_non_crisis_df)
    
    # 5. Load custom crisis phrases (optional)
    custom_df = load_custom_crisis_phrases()
    if not custom_df.empty:
        all_datasets.append(custom_df)
    
    if not all_datasets:
        print("\n  [ERROR] No datasets loaded! Please check dataset paths.")
        return
    
    # Combine all datasets
    print("\n[4] Combining datasets...")
    combined_df = pd.concat(all_datasets, ignore_index=True)
    print(f"  [OK] Total combined samples: {len(combined_df)}")
    
    # Remove duplicates
    print("\n[5] Removing duplicates...")
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
    after_dedup = len(combined_df)
    print(f"  [OK] Removed {before_dedup - after_dedup} duplicates")
    print(f"  Remaining samples: {after_dedup}")
    
    # Clean: remove empty texts
    combined_df = combined_df[combined_df['text'].astype(str).str.strip() != '']
    
    # Show distribution before splitting
    print("\n[6] Dataset distribution:")
    crisis_count = (combined_df['label'] == 1).sum()
    non_crisis_count = (combined_df['label'] == 0).sum()
    print(f"  Crisis: {crisis_count} ({crisis_count/len(combined_df)*100:.1f}%)")
    print(f"  Non-crisis: {non_crisis_count} ({non_crisis_count/len(combined_df)*100:.1f}%)")
    
    # Split into train/validation/test (80/10/10)
    print("\n[7] Splitting dataset (80% train, 10% validation, 10% test)...")
    train_df, val_df, test_df = split_dataset(combined_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42)
    
    print(f"  [OK] Train: {len(train_df)} samples")
    print(f"    Crisis: {(train_df['label'] == 1).sum()}, Non-crisis: {(train_df['label'] == 0).sum()}")
    print(f"  [OK] Validation: {len(val_df)} samples")
    print(f"    Crisis: {(val_df['label'] == 1).sum()}, Non-crisis: {(val_df['label'] == 0).sum()}")
    print(f"  [OK] Test: {len(test_df)} samples")
    print(f"    Crisis: {(test_df['label'] == 1).sum()}, Non-crisis: {(test_df['label'] == 0).sum()}")
    
    # Balance each split separately
    print("\n[8] Balancing splits...")
    train_balanced = balance_dataset(train_df, target_ratio=0.5, random_state=42)
    val_balanced = balance_dataset(val_df, target_ratio=0.5, random_state=42)
    test_balanced = balance_dataset(test_df, target_ratio=0.5, random_state=42)
    
    all_data = {
        'train': train_balanced,
        'validation': val_balanced,
        'test': test_balanced
    }
    
    print(f"  [OK] Balanced train: {len(train_balanced)} samples")
    print(f"  [OK] Balanced validation: {len(val_balanced)} samples")
    print(f"  [OK] Balanced test: {len(test_balanced)} samples")
    
    # Save processed data
    print("\n[9] Saving processed data...")
    
    for split in ['train', 'validation', 'test']:
        # Use 'valid' for validation to match convention
        filename = 'valid.csv' if split == 'validation' else f"{split}.csv"
        output_file = OUTPUT_DIR / filename
        
        df = all_data[split]
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Also save as validation.csv if needed
        if split == 'validation':
            df.to_csv(OUTPUT_DIR / 'validation.csv', index=False, encoding='utf-8')
        
        print(f"  [OK] Saved {filename}: {len(df)} rows")
        
        # Show distribution
        if not df.empty:
            crisis_count = (df['label'] == 1).sum()
            non_crisis_count = (df['label'] == 0).sum()
            print(f"    Crisis: {crisis_count}, Non-crisis: {non_crisis_count}")
    
    print("\n" + "=" * 70)
    print("Preprocessing completed!")
    print(f"Processed data saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
