"""
Training script for emotion classification model - Google Colab Optimized.

This script:
- Loads processed train.csv and validation.csv
- Loads label_map.json for label mapping
- Converts pipe-separated labels into multi-hot vectors
- Tokenizes text using distilbert-base-uncased
- Trains the model with GPU optimizations
- Saves model to Google Drive or local directory
"""

import pandas as pd
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from pathlib import Path
from typing import List, Dict, Any
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ===== COLAB CONFIGURATION =====
# Set this to True if running in Google Colab
RUNNING_IN_COLAB = os.path.exists('/content')

if RUNNING_IN_COLAB:
    # Colab paths
    BASE_DIR = Path("/content/emotion_model")
    # Optional: Use Google Drive for saving
    USE_DRIVE = True  # Set to False to save locally
    if USE_DRIVE:
        DRIVE_MODEL_DIR = Path("/content/drive/MyDrive/sehatmind/chatbot_backend/emotion_model/model")
    else:
        DRIVE_MODEL_DIR = None
else:
    # Local paths
    BASE_DIR = Path(__file__).parent

PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"
MODEL_OUTPUT_DIR = BASE_DIR / "model"

# Model configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

# GPU optimizations
USE_FP16 = True  # Mixed precision training (faster on GPU)
BATCH_SIZE = 32 if torch.cuda.is_available() else 16  # Larger batch on GPU


class EmotionDataset(Dataset):
    """
    Dataset class for emotion classification.
    
    Handles:
    - Text tokenization
    - Multi-label encoding (pipe-separated labels to multi-hot vectors)
    - Padding and truncation
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer: AutoTokenizer,
        label_map: Dict[str, Any],
        max_length: int = 128
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            labels: List of label strings (pipe-separated for multi-label)
            tokenizer: Tokenizer instance
            label_map: Label mapping dictionary
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
        
        # Get label to ID mapping
        self.label_to_id = label_map["label_to_id"]
        self.num_labels = label_map["num_labels"]
    
    def __len__(self):
        """Return the number of examples."""
        return len(self.texts)
    
    def labels_to_multihot(self, label_string: str) -> np.ndarray:
        """
        Convert pipe-separated labels to multi-hot vector.
        
        Example:
            "joy|sadness" -> [1, 1, 0, 0, 0, 0, 0]
            "anger" -> [0, 0, 1, 0, 0, 0, 0]
        
        Args:
            label_string: Pipe-separated label string (e.g., "joy|sadness")
        
        Returns:
            Multi-hot vector as numpy array
        """
        # Initialize multi-hot vector with zeros
        multihot = np.zeros(self.num_labels, dtype=np.float32)
        
        if pd.isna(label_string) or label_string == '':
            return multihot
        
        # Split by pipe separator
        labels = str(label_string).split('|')
        
        # Set corresponding indices to 1
        for label in labels:
            label = label.strip()
            if label in self.label_to_id:
                label_id = self.label_to_id[label]
                multihot[label_id] = 1.0
        
        return multihot
    
    def __getitem__(self, idx):
        """
        Get a single example.
        
        Args:
            idx: Index of the example
        
        Returns:
            Dictionary with tokenized inputs and labels
        """
        text = str(self.texts[idx])
        label_string = str(self.labels[idx]) if not pd.isna(self.labels[idx]) else ""
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert labels to multi-hot vector
        labels = self.labels_to_multihot(label_string)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }


def load_label_map(label_map_path: Path) -> Dict[str, Any]:
    """
    Load the label mapping from JSON file.
    
    Args:
        label_map_path: Path to label_map.json
    
    Returns:
        Label mapping dictionary
    """
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    return label_map


def load_processed_data(data_dir: Path, split: str) -> pd.DataFrame:
    """
    Load processed CSV file.
    
    Args:
        data_dir: Directory containing processed data
        split: Data split name ('train' or 'validation')
    
    Returns:
        DataFrame with 'text' and 'labels' columns
    """
    filepath = data_dir / f"{split}.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data file not found: {filepath}")
    
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Validate required columns
    if 'text' not in df.columns or 'labels' not in df.columns:
        raise ValueError(f"CSV file must contain 'text' and 'labels' columns")
    
    print(f"  [OK] Loaded {split}.csv: {len(df)} rows")
    
    return df


def create_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    label_map: Dict[str, Any]
) -> tuple[EmotionDataset, EmotionDataset]:
    """
    Create datasets for training and validation.
    
    Args:
        train_df: Training DataFrame
        validation_df: Validation DataFrame
        tokenizer: Tokenizer instance
        label_map: Label mapping dictionary
    
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    # Create datasets
    train_dataset = EmotionDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['labels'].tolist(),
        tokenizer=tokenizer,
        label_map=label_map,
        max_length=MAX_LENGTH
    )
    
    validation_dataset = EmotionDataset(
        texts=validation_df['text'].tolist(),
        labels=validation_df['labels'].tolist(),
        tokenizer=tokenizer,
        label_map=label_map,
        max_length=MAX_LENGTH
    )
    
    print(f"  [OK] Created training dataset: {len(train_dataset)} examples")
    print(f"  [OK] Created validation dataset: {len(validation_dataset)} examples")
    
    return train_dataset, validation_dataset


def compute_metrics(eval_pred):
    """
    Compute metrics for multi-label classification.
    
    Args:
        eval_pred: Tuple of (predictions, labels) from the model
    
    Returns:
        Dictionary with computed metrics
    """
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    
    # Calculate metrics
    # For multi-label, we calculate per-sample accuracy and then average
    accuracy = accuracy_score(labels, y_pred)
    
    # Macro-averaged F1 (average F1 across all labels)
    f1_macro = f1_score(labels, y_pred, average='macro', zero_division=0)
    
    # Micro-averaged F1 (F1 calculated globally)
    f1_micro = f1_score(labels, y_pred, average='micro', zero_division=0)
    
    # Weighted F1 (F1 averaged by support)
    f1_weighted = f1_score(labels, y_pred, average='weighted', zero_division=0)
    
    # Precision and Recall (macro-averaged)
    precision = precision_score(labels, y_pred, average='macro', zero_division=0)
    recall = recall_score(labels, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }


def main():
    """Main training function."""
    print("=" * 70)
    print("Emotion Classification Model - Colab Optimized Training")
    print("=" * 70)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n[OK] GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n[WARN] No GPU available, training will be slow on CPU")
    
    # Load label map
    print("\n[1] Loading label mapping...")
    label_map = load_label_map(LABEL_MAP_PATH)
    print(f"  [OK] Loaded label map with {label_map['num_labels']} emotions")
    print(f"    Emotions: {list(label_map['label_to_id'].keys())}")
    
    # Load processed data
    print("\n[2] Loading processed data...")
    train_df = load_processed_data(PROCESSED_DATA_DIR, 'train')
    validation_df = load_processed_data(PROCESSED_DATA_DIR, 'validation')
    
    # Load tokenizer
    print("\n[3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  [OK] Loaded tokenizer: {MODEL_NAME}")
    
    # Create datasets
    print("\n[4] Creating datasets...")
    train_dataset, validation_dataset = create_datasets(
        train_df=train_df,
        validation_df=validation_df,
        tokenizer=tokenizer,
        label_map=label_map
    )
    
    # Load model
    print("\n[5] Loading model...")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=label_map['num_labels'],
        problem_type="multi_label_classification"
    )
    print(f"  [OK] Loaded model: {MODEL_NAME}")
    print(f"    Number of labels: {label_map['num_labels']}")
    print(f"    Problem type: multi_label_classification")
    
    # Configure training arguments
    print("\n[6] Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        eval_strategy="epoch",
        learning_rate=3e-5,  # Slightly higher LR for better learning
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Larger eval batch for faster validation
        num_train_epochs=5,  # More epochs for better learning
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",  # FIXED: Matches compute_metrics return key
        greater_is_better=True,
        logging_dir=str(MODEL_OUTPUT_DIR / "logs"),
        logging_steps=100,
        save_total_limit=3,  # Keep more checkpoints
        warmup_steps=300,  # Reduced warmup steps
        weight_decay=0.01,
        gradient_accumulation_steps=1,  # Can increase if running out of memory
        report_to="none",
        fp16=USE_FP16 and torch.cuda.is_available(),  # Mixed precision for GPU
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,  # Parallel data loading
        save_steps=500  # Save checkpoint every N steps as backup
    )
    print(f"  [OK] Training arguments configured")
    print(f"    Learning rate: {training_args.learning_rate}")
    print(f"    Train batch size: {training_args.per_device_train_batch_size}")
    print(f"    Eval batch size: {training_args.per_device_eval_batch_size}")
    print(f"    Epochs: {training_args.num_train_epochs}")
    print(f"    Evaluation strategy: {training_args.eval_strategy}")
    print(f"    Best model metric: {training_args.metric_for_best_model}")
    print(f"    Warmup steps: {training_args.warmup_steps}")
    print(f"    FP16 (mixed precision): {training_args.fp16}")
    print(f"    Output directory: {training_args.output_dir}")
    
    # Create Trainer
    print("\n[7] Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics
    )
    print(f"  [OK] Trainer created")
    
    # Train the model
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print(f"Training examples: {len(train_df)}")
    print(f"Validation examples: {len(validation_df)}")
    print(f"Number of emotion classes: {label_map['num_labels']}")
    print(f"\nTraining will run for {training_args.num_train_epochs} epochs...")
    print("This may take a while depending on your hardware.\n")
    
    try:
        # Train the model
        train_result = trainer.train()
        
        print("\n" + "=" * 70)
        print("Training Completed!")
        print("=" * 70)
        print(f"Training loss: {train_result.training_loss:.4f}")
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        eval_results = trainer.evaluate()
        
        print("\nValidation Results:")
        print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"  F1 Macro: {eval_results['eval_f1_macro']:.4f}")
        print(f"  F1 Micro: {eval_results['eval_f1_micro']:.4f}")
        print(f"  F1 Weighted: {eval_results['eval_f1_weighted']:.4f}")
        print(f"  Precision: {eval_results['eval_precision']:.4f}")
        print(f"  Recall: {eval_results['eval_recall']:.4f}")
        
        # Save the final model and tokenizer
        print(f"\n[8] Saving model and tokenizer...")
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(str(MODEL_OUTPUT_DIR))
        tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
        
        # Save label map for inference
        with open(MODEL_OUTPUT_DIR / "label_map.json", 'w', encoding='utf-8') as f:
            json.dump(label_map, f, indent=2)
        
        print(f"  [OK] Model saved to: {MODEL_OUTPUT_DIR}")
        print(f"  [OK] Tokenizer saved")
        print(f"  [OK] Label map saved")
        
        # Also save to Google Drive if enabled
        if RUNNING_IN_COLAB and USE_DRIVE and DRIVE_MODEL_DIR:
            print(f"\n[9] Saving to Google Drive...")
            DRIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            # Copy model files to Drive
            import shutil
            if MODEL_OUTPUT_DIR.exists():
                shutil.copytree(MODEL_OUTPUT_DIR, DRIVE_MODEL_DIR, dirs_exist_ok=True)
                print(f"  [OK] Model saved to Google Drive: {DRIVE_MODEL_DIR}")
        
        print("\n" + "=" * 70)
        print("Training and saving completed successfully!")
        print("=" * 70)
        print(f"\nModel is ready for use at: {MODEL_OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

