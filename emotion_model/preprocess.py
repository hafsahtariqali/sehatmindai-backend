"""
Preprocessing script for emotion datasets.

This script:
- Loads train/valid/test CSV files from chatbot_backend/datasets
- Supports GoEmotions, DailyDialog, and Reddit Emotions datasets
- Standardizes column names to 'text' and 'labels'
- Converts numeric labels to emotion names using label_map.json
- Handles multi-label rows (converts to pipe-separated labels)
- Saves processed data to emotion_model/processed_data/
"""

import pandas as pd
import json
import os
import ast
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


# Define paths
BASE_DIR = Path(__file__).parent.parent  # chatbot_backend/
DATASETS_DIR = BASE_DIR / "datasets"
LABEL_MAP_PATH = Path(__file__).parent / "label_map.json"
OUTPUT_DIR = Path(__file__).parent / "processed_data"

# Supported datasets
SUPPORTED_DATASETS = ["goemotions", "dailydialog", "reddit_emotions"]


def load_label_map(label_map_path: Path) -> Dict[str, Any]:
    """
    Load the unified emotion label mapping.
    
    Args:
        label_map_path: Path to label_map.json
    
    Returns:
        Dictionary with label mappings
    """
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    return label_map


def find_dataset_files(dataset_name: str, split: str) -> Optional[Path]:
    """
    Find dataset files for a given dataset and split.
    
    Looks for files with patterns like:
    - {dataset_name}_{split}.csv
    - {split}.csv
    - train.csv, validation.csv, test.csv
    - Also checks for "valid" as alternative to "validation"
    
    Args:
        dataset_name: Name of the dataset folder
        split: Data split (train, validation, test)
    
    Returns:
        Path to the file if found, None otherwise
    """
    dataset_dir = DATASETS_DIR / dataset_name
    
    if not dataset_dir.exists():
        return None
    
    # Try different naming patterns
    # If split is "validation", also try "valid"
    split_variants = [split]
    if split == "validation":
        split_variants.append("valid")
    elif split == "valid":
        split_variants.append("validation")
    
    patterns = []
    for variant in split_variants:
        patterns.extend([
            f"{dataset_name}_{variant}.csv",
            f"{variant}.csv",
            f"{variant}.tsv",
            f"{variant}_data.csv",
        ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_patterns = []
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique_patterns.append(pattern)
    
    for pattern in unique_patterns:
        filepath = dataset_dir / pattern
        if filepath.exists():
            return filepath
    
    # Look for any CSV file in the directory
    csv_files = list(dataset_dir.glob("*.csv"))
    if csv_files and split == "train":
        return csv_files[0]  # Return first CSV as fallback
    
    return None


def normalize_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize text column name to 'text'.
    
    Common column names: text, sentence, utterance, comment, post, message
    
    Args:
        df: DataFrame to normalize
    
    Returns:
        DataFrame with 'text' column
    """
    text_columns = ['text', 'sentence', 'utterance', 'comment', 'post', 'message', 
                   'input', 'context', 'dialogue', 'query']
    
    for col in text_columns:
        if col in df.columns:
            df = df.rename(columns={col: 'text'})
            break
    
    if 'text' not in df.columns and len(df.columns) > 0:
        # Use first column as text if no standard column found
        df = df.rename(columns={df.columns[0]: 'text'})
    
    return df


def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize label column name to 'labels'.
    
    Common column names: label, labels, emotion, emotion_label, category, class
    
    Args:
        df: DataFrame to normalize
    
    Returns:
        DataFrame with 'labels' column
    """
    # Priority order: prefer numeric emotion IDs, then text emotion labels
    label_columns = ['labels', 'label', 'emotion', 'emotion_id', 'category', 
                    'class', 'sentiment', 'matched_emotions', 'emotion_label']
    
    # For DailyDialog: prefer 'emotion' (numeric) over 'emotion_label' (text)
    # For Reddit Emotions: use 'matched_emotions' (text list)
    # Check for specific dataset patterns
    if 'matched_emotions' in df.columns:
        # Reddit Emotions - use matched_emotions
        df = df.rename(columns={'matched_emotions': 'labels'})
    elif 'emotion' in df.columns and 'emotion_label' in df.columns:
        # DailyDialog - prefer numeric 'emotion' over text 'emotion_label'
        df = df.rename(columns={'emotion': 'labels'})
        # Drop emotion_label column to avoid confusion
        df = df.drop(columns=['emotion_label'], errors='ignore')
    else:
        # Standard column matching
        for col in label_columns:
            if col in df.columns:
                df = df.rename(columns={col: 'labels'})
                break
    
    return df


def convert_numeric_labels_to_emotions(
    df: pd.DataFrame,
    label_map: Dict[str, Any],
    dataset_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert numeric labels to emotion names using label_map.
    
    Handles:
    - String representations of lists: "[27]" or "[8, 20]"
    - Direct numeric IDs: 2, 14, etc.
    - Emotion names (already converted): "joy", "sadness"
    - Multi-label lists: [0, 1, 2]
    
    Args:
        df: DataFrame with 'labels' column containing numeric IDs
        label_map: Label mapping dictionary
    
    Returns:
        DataFrame with emotion names in 'labels' column
    """
    df = df.copy()
    id_to_label = label_map["id_to_label"]
    
    # Try to load dataset-specific mapping if available
    dataset_mapping_path = Path(__file__).parent / "dataset_label_mapping.json"
    dataset_mappings = {}
    if dataset_mapping_path.exists():
        try:
            with open(dataset_mapping_path, 'r', encoding='utf-8') as f:
                dataset_mappings = json.load(f)
        except:
            pass
    
    def convert_label(label_value):
        """Convert a single label value."""
        if pd.isna(label_value):
            return None
        
        # Convert to string for processing
        label_str = str(label_value).strip().lower()  # Convert to lowercase for matching
        
        # Check if it's already an emotion name (pipe-separated or single)
        if '|' in label_str:
            # Already pipe-separated emotion names - validate and return
            labels = [l.strip().lower() for l in label_str.split('|')]
            valid_labels = [l for l in labels if l in label_map["label_to_id"]]
            return '|'.join(valid_labels) if valid_labels else None
        
        # Check if it's already a valid emotion name (case-insensitive)
        if label_str in label_map["label_to_id"]:
            return label_str
        
        # FIRST: Try text-based emotion name mapping (for Reddit Emotions, etc.)
        # Check if it's a list of emotion names (like "['anxious', 'lonely']")
        emotion_names_from_text = []
        try:
            parsed = ast.literal_eval(label_str)
            if isinstance(parsed, list):
                # List of emotion names (Reddit Emotions format)
                for item in parsed:
                    item_str = str(item).strip().lower()
                    # Try direct mapping (already valid emotion name)
                    if item_str in label_map["label_to_id"]:
                        emotion_names_from_text.append(item_str)
                    # Try dataset-specific text mapping (reddit_emotions)
                    elif "reddit_emotions" in dataset_mappings and item_str in dataset_mappings["reddit_emotions"]:
                        mapped = dataset_mappings["reddit_emotions"][item_str]
                        if mapped in label_map["label_to_id"]:
                            emotion_names_from_text.append(mapped)
            elif isinstance(parsed, str):
                # Single emotion name string
                parsed_lower = parsed.strip().lower()
                if parsed_lower in label_map["label_to_id"]:
                    emotion_names_from_text.append(parsed_lower)
                elif "reddit_emotions" in dataset_mappings and parsed_lower in dataset_mappings["reddit_emotions"]:
                    mapped = dataset_mappings["reddit_emotions"][parsed_lower]
                    if mapped in label_map["label_to_id"]:
                        emotion_names_from_text.append(mapped)
        except (ValueError, SyntaxError):
            # Not a list, try as single text emotion name
            label_str_lower = label_str.lower()
            if label_str_lower in label_map["label_to_id"]:
                emotion_names_from_text.append(label_str_lower)
            elif "reddit_emotions" in dataset_mappings and label_str_lower in dataset_mappings["reddit_emotions"]:
                mapped = dataset_mappings["reddit_emotions"][label_str_lower]
                if mapped in label_map["label_to_id"]:
                    emotion_names_from_text.append(mapped)
        
        if emotion_names_from_text:
            return '|'.join(sorted(set(emotion_names_from_text)))
        
        # SECOND: Try numeric ID mapping (for GoEmotions, DailyDialog)
        label_ids = []
        try:
            # Try to parse as Python literal (list of numbers)
            parsed = ast.literal_eval(label_str)
            if isinstance(parsed, list):
                label_ids = [int(x) for x in parsed if isinstance(x, (int, float))]
            elif isinstance(parsed, (int, float)):
                label_ids = [int(parsed)]
        except (ValueError, SyntaxError):
            # Not a list format, try direct integer conversion
            try:
                label_int = int(float(label_str))
                label_ids = [label_int]
            except (ValueError, TypeError):
                pass
        
        # Convert numeric IDs to emotion names
        emotion_names = []
        for label_id in label_ids:
            label_id_str = str(label_id)
            
            # First try dataset-specific mapping if dataset_name is known
            if dataset_name and dataset_name in dataset_mappings:
                if label_id_str in dataset_mappings[dataset_name]:
                    mapped_emotion = dataset_mappings[dataset_name][label_id_str]
                    if mapped_emotion in label_map["label_to_id"]:
                        emotion_names.append(mapped_emotion)
                        continue
            
            # Fallback: Try direct mapping (0-6)
            if label_id_str in id_to_label:
                emotion_names.append(id_to_label[label_id_str])
            else:
                # Try other dataset mappings as fallback (excluding text-based mappings)
                for map_ds_name, mapping in dataset_mappings.items():
                    if map_ds_name not in ["note", "reddit_emotions", "goemotions_labels", "dailydialog_labels"]:
                        if label_id_str in mapping:
                            mapped_emotion = mapping[label_id_str]
                            if mapped_emotion in label_map["label_to_id"]:
                                emotion_names.append(mapped_emotion)
                                break
        
        # Return pipe-separated emotion names
        if emotion_names:
            return '|'.join(sorted(set(emotion_names)))
        
        # If no mapping found, return None (will be filtered out later)
        return None
    
    # Apply conversion
    if 'labels' in df.columns:
        df['labels'] = df['labels'].apply(convert_label)
        
        # Filter out rows with None labels (unmappable labels)
        initial_count = len(df)
        df = df[df['labels'].notna()].copy()
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"    Filtered out {filtered_count} rows with unmappable labels")
    
    return df


def handle_multi_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert multi-label rows to pipe-separated format.
    
    Handles cases where:
    - Multiple labels are in a list
    - Multiple labels are in separate columns
    - Labels are already pipe-separated
    
    Args:
        df: DataFrame with 'labels' column
    
    Returns:
        DataFrame with pipe-separated labels
    """
    df = df.copy()
    
    if 'labels' not in df.columns:
        return df
    
    def process_labels(label_value):
        """Process a single label value to pipe-separated format."""
        if pd.isna(label_value):
            return None
        
        # If it's already a string with pipes, return as is
        if isinstance(label_value, str) and '|' in label_value:
            # Clean up and validate
            labels = [l.strip() for l in label_value.split('|') if l.strip()]
            return '|'.join(labels) if labels else None
        
        # If it's a list, join with pipe
        if isinstance(label_value, list):
            labels = [str(l).strip() for l in label_value if pd.notna(l)]
            return '|'.join(labels) if labels else None
        
        # If it's a single value, return as string
        return str(label_value).strip()
    
    df['labels'] = df['labels'].apply(process_labels)
    
    # Check for multiple label columns (one-hot encoded labels)
    label_columns = [col for col in df.columns 
                    if col not in ['text', 'labels'] 
                    and col.lower() in ['joy', 'sadness', 'anger', 'fear', 
                                      'anxiety', 'loneliness', 'neutral']]
    
    if label_columns and len(label_columns) > 0:
        # Convert one-hot encoded labels to pipe-separated
        def combine_one_hot(row):
            """Combine one-hot encoded labels."""
            active_labels = []
            for col in label_columns:
                if col in row and (row[col] == 1 or row[col] == True or 
                                 str(row[col]).lower() == 'true'):
                    active_labels.append(col)
            return '|'.join(active_labels) if active_labels else None
        
        # Only use one-hot if labels column is missing or empty
        if df['labels'].isna().all() or df['labels'].eq('').all():
            df['labels'] = df.apply(combine_one_hot, axis=1)
            # Drop the one-hot columns after combining
            df = df.drop(columns=label_columns)
    
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the DataFrame.
    
    - Keep only 'text' and 'labels' columns
    - Remove rows with missing text
    - Remove rows with missing labels (optional - might want to keep for some cases)
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Keep only text and labels columns
    columns_to_keep = ['text', 'labels']
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]
    
    # Remove rows with missing text
    df = df.dropna(subset=['text'])
    df = df[df['text'].astype(str).str.strip() != '']
    
    # Remove rows with missing labels (uncomment if needed)
    # df = df.dropna(subset=['labels'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def load_and_process_dataset(
    dataset_name: str,
    split: str,
    label_map: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """
    Load and process a single dataset file.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split (train, validation, test)
        label_map: Label mapping dictionary
    
    Returns:
        Processed DataFrame or None if file not found
    """
    # Find the dataset file
    filepath = find_dataset_files(dataset_name, split)
    
    if filepath is None:
        print(f"  ⚠ File not found for {dataset_name}/{split}")
        return None
    
    print(f"  Loading {filepath.name}...")
    
    try:
        # Load CSV
        # Try different separators
        try:
            df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
        except:
            try:
                df = pd.read_csv(filepath, sep='\t', encoding='utf-8', on_bad_lines='skip')
            except:
                df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip')
        
        if df.empty:
            print(f"  ⚠ Empty file: {filepath.name}")
            return None
        
        print(f"    Loaded {len(df)} rows")
        
        # Normalize columns
        df = normalize_text_column(df)
        df = normalize_label_column(df)
        
        # Convert numeric labels to emotion names (pass dataset_name for correct mapping)
        df = convert_numeric_labels_to_emotions(df, label_map, dataset_name=dataset_name)
        
        # Handle multi-label rows
        df = handle_multi_label(df)
        
        # Clean DataFrame
        df = clean_dataframe(df)
        
        print(f"    Processed {len(df)} rows")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error processing {filepath.name}: {e}")
        return None


def combine_splits(all_data: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Combine multiple datasets for each split.
    
    Args:
        all_data: Dictionary mapping split names to lists of DataFrames
    
    Returns:
        Dictionary mapping split names to combined DataFrames
    """
    combined = {}
    
    for split, dataframes in all_data.items():
        if not dataframes:
            print(f"  ⚠ No data for {split} split")
            combined[split] = pd.DataFrame(columns=['text', 'labels'])
        else:
            combined_df = pd.concat(dataframes, ignore_index=True)
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
            combined_df = combined_df.reset_index(drop=True)
            combined[split] = combined_df
            print(f"  ✓ Combined {split}: {len(combined_df)} rows")
    
    return combined


def main():
    """Main preprocessing function."""
    print("=" * 70)
    print("Emotion Dataset Preprocessing")
    print("=" * 70)
    
    # Load label map
    print("\n[1] Loading label mapping...")
    label_map = load_label_map(LABEL_MAP_PATH)
    print(f"  ✓ Loaded label map with {label_map['num_labels']} emotions")
    print(f"    Emotions: {list(label_map['label_to_id'].keys())}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[2] Output directory: {OUTPUT_DIR}")
    
    # Process all datasets
    print("\n[3] Processing datasets...")
    all_data = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    for dataset_name in SUPPORTED_DATASETS:
        print(f"\n  Processing {dataset_name}...")
        
        for split in ['train', 'validation', 'test']:
            df = load_and_process_dataset(dataset_name, split, label_map)
            if df is not None and not df.empty:
                all_data[split].append(df)
    
    # Combine splits across all datasets
    print("\n[4] Combining datasets...")
    combined_data = combine_splits(all_data)
    
    # Save processed data
    print("\n[5] Saving processed data...")
    
    for split in ['train', 'validation', 'test']:
        output_file = OUTPUT_DIR / f"{split}.csv"
        combined_data[split].to_csv(output_file, index=False, encoding='utf-8')
        print(f"  ✓ Saved {split}.csv: {len(combined_data[split])} rows")
    
    print("\n" + "=" * 70)
    print("Preprocessing completed!")
    print(f"Processed data saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

