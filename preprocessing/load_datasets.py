"""
Dataset Loading and Cleaning Script

This script loads datasets from the datasets folder, normalizes their columns,
and saves cleaned versions as CSV files in the preprocessing/cleaned/ folder.

Each dataset may have train, validation, and test splits - these are kept separate.
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


# Define the base directories
# This script assumes it's run from the chatbot_backend directory
BASE_DIR = Path(__file__).parent.parent  # chatbot_backend/
DATASETS_DIR = BASE_DIR / "datasets"
CLEANED_DIR = BASE_DIR / "preprocessing" / "cleaned"


def create_output_dir(dataset_name: str) -> Path:
    """
    Create output directory for a specific dataset if it doesn't exist.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'empathetic_dialogue')
    
    Returns:
        Path object for the output directory
    """
    output_dir = CLEANED_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def detect_file_type(file_path: Path) -> str:
    """
    Detect the file type based on extension.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File extension (e.g., 'csv', 'json', 'txt')
    """
    return file_path.suffix.lower().lstrip('.')


def detect_split_type(filename: str) -> Optional[str]:
    """
    Detect if a file is train, validation, or test based on filename.
    
    Common patterns: train, val, validation, dev, test, eval
    
    Args:
        filename: Name of the file (without path)
    
    Returns:
        Split type ('train', 'val', or 'test') or None if not detected
    """
    filename_lower = filename.lower()
    
    # Check for train split
    if 'train' in filename_lower:
        return 'train'
    
    # Check for validation/dev split
    if any(keyword in filename_lower for keyword in ['val', 'validation', 'dev']):
        return 'val'
    
    # Check for test split
    if 'test' in filename_lower or 'eval' in filename_lower:
        return 'test'
    
    return None


def normalize_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Normalize column names to standard format: 'text' and optionally 'label'.
    
    This function tries to map various column names to 'text' and 'label'.
    Common text column names: utterance, dialogue, text, sentence, message, response
    Common label column names: emotion, label, sentiment, category, class
    
    Args:
        df: DataFrame to normalize
        dataset_name: Name of the dataset (for logging)
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    columns_lower = {col.lower(): col for col in df.columns}
    
    # Map text columns - try common variations
    text_keywords = ['text', 'utterance', 'dialogue', 'sentence', 'message', 
                     'response', 'context', 'input', 'query', 'statement']
    text_column = None
    
    for keyword in text_keywords:
        if keyword in columns_lower:
            text_column = columns_lower[keyword]
            break
    
    # If no text column found, use the first column as fallback
    if text_column is None:
        text_column = df.columns[0]
        print(f"  Warning: No standard text column found in {dataset_name}. Using '{text_column}' as text.")
    
    # Rename text column
    df_normalized = df_normalized.rename(columns={text_column: 'text'})
    
    # Map label columns - try common variations
    label_keywords = ['label', 'emotion', 'sentiment', 'category', 'class', 
                      'intent', 'emotion_label', 'target']
    label_column = None
    
    for keyword in label_keywords:
        if keyword in columns_lower:
            label_column = columns_lower[keyword]
            break
    
    # Rename label column if found
    if label_column and label_column != text_column:
        df_normalized = df_normalized.rename(columns={label_column: 'label'})
        print(f"  Found label column: '{label_column}' -> 'label'")
    else:
        print(f"  No label column found in {dataset_name}.")
    
    # Keep only text and label columns (if label exists)
    columns_to_keep = ['text']
    if 'label' in df_normalized.columns:
        columns_to_keep.append('label')
    
    # Also keep any columns that are already named 'text' or 'label'
    df_normalized = df_normalized[[col for col in df_normalized.columns 
                                   if col in columns_to_keep or col.lower() in ['text', 'label']]]
    
    return df_normalized


def load_csv_file(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        DataFrame containing the data
    """
    try:
        # Try different separators (comma, tab, semicolon)
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        if df.empty or len(df.columns) == 1:
            # If only one column, try tab separator
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        print(f"  Error loading CSV {file_path.name}: {e}")
        return pd.DataFrame()
    
    return df


def load_json_file(file_path: Path) -> pd.DataFrame:
    """
    Load a JSON file into a pandas DataFrame.
    
    Handles both JSON arrays and JSONL (JSON Lines) formats.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        DataFrame containing the data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if it's JSONL (one JSON object per line)
            first_line = f.readline().strip()
            f.seek(0)  # Reset file pointer
            
            if first_line.startswith('['):
                # JSON array format
                data = json.load(f)
                df = pd.DataFrame(data)
            else:
                # JSONL format (one JSON object per line)
                data = []
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
                df = pd.DataFrame(data)
        
        return df
    except Exception as e:
        print(f"  Error loading JSON {file_path.name}: {e}")
        return pd.DataFrame()


def load_txt_file(file_path: Path) -> pd.DataFrame:
    """
    Load a text file into a pandas DataFrame.
    
    Assumes one entry per line. Creates a 'text' column.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        DataFrame with a 'text' column
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        df = pd.DataFrame({'text': lines})
        return df
    except Exception as e:
        print(f"  Error loading TXT {file_path.name}: {e}")
        return pd.DataFrame()


def load_dataset_file(file_path: Path) -> pd.DataFrame:
    """
    Load a dataset file based on its extension.
    
    Supports CSV, JSON, JSONL, and TXT formats.
    
    Args:
        file_path: Path to the file
    
    Returns:
        DataFrame containing the loaded data
    """
    file_type = detect_file_type(file_path)
    
    if file_type == 'csv':
        return load_csv_file(file_path)
    elif file_type == 'json':
        return load_json_file(file_path)
    elif file_type == 'txt' or file_type == 'text':
        return load_txt_file(file_path)
    else:
        print(f"  Unsupported file type: {file_type} for {file_path.name}")
        return pd.DataFrame()


def process_dataset(dataset_name: str) -> None:
    """
    Process all files in a dataset folder.
    
    This function:
    1. Finds all files in the dataset folder
    2. Detects train/val/test splits
    3. Loads each file
    4. Normalizes columns
    5. Saves cleaned CSV files
    
    Args:
        dataset_name: Name of the dataset folder
    """
    dataset_dir = DATASETS_DIR / dataset_name
    
    if not dataset_dir.exists():
        print(f"Dataset folder '{dataset_name}' not found. Skipping...")
        return
    
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"  Location: {dataset_dir}")
    
    # Get all files in the dataset directory
    files = list(dataset_dir.glob('*'))
    files = [f for f in files if f.is_file() and not f.name.startswith('.')]
    
    if not files:
        print(f"  No files found in {dataset_name}")
        return
    
    # Create output directory for this dataset
    output_dir = create_output_dir(dataset_name)
    print(f"  Output directory: {output_dir}")
    
    # Process each file
    for file_path in files:
        print(f"\n  Processing file: {file_path.name}")
        
        # Detect split type (train/val/test)
        split_type = detect_split_type(file_path.name)
        if split_type:
            print(f"  Detected split: {split_type}")
        else:
            print(f"  Split type: unknown (using filename)")
        
        # Load the file
        df = load_dataset_file(file_path)
        
        if df.empty:
            print(f"  Warning: File {file_path.name} is empty or could not be loaded.")
            continue
        
        print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"  Original columns: {list(df.columns)}")
        
        # Normalize columns
        df_normalized = normalize_columns(df, dataset_name)
        print(f"  Normalized columns: {list(df_normalized.columns)}")
        
        # Create output filename
        if split_type:
            output_filename = f"{dataset_name}_{split_type}.csv"
        else:
            # Use original filename without extension
            output_filename = f"{dataset_name}_{file_path.stem}.csv"
        
        output_path = output_dir / output_filename
        
        # Save cleaned CSV
        df_normalized.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  Saved cleaned file: {output_path}")
        print(f"  Rows saved: {len(df_normalized)}")


def main():
    """
    Main function that processes all datasets.
    
    This function:
    1. Checks if the datasets directory exists
    2. Gets list of all dataset folders
    3. Processes each dataset
    4. Prints summary
    """
    print("=" * 60)
    print("Dataset Loading and Cleaning Script")
    print("=" * 60)
    
    # Check if datasets directory exists
    if not DATASETS_DIR.exists():
        print(f"Error: Datasets directory not found at {DATASETS_DIR}")
        print("Please ensure the datasets folder exists.")
        return
    
    # Create cleaned output directory if it doesn't exist
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all dataset folders
    dataset_folders = [d.name for d in DATASETS_DIR.iterdir() 
                      if d.is_dir() and not d.name.startswith('.')]
    
    if not dataset_folders:
        print(f"No dataset folders found in {DATASETS_DIR}")
        print("Please add dataset folders to the datasets directory.")
        return
    
    print(f"\nFound {len(dataset_folders)} dataset folder(s): {', '.join(dataset_folders)}")
    
    # Process each dataset
    for dataset_name in sorted(dataset_folders):
        process_dataset(dataset_name)
    
    print("\n" + "=" * 60)
    print("Dataset loading and cleaning completed!")
    print(f"Cleaned files saved to: {CLEANED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    # Run the main function when script is executed
    main()

