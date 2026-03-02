"""
Training script for crisis detection model - Google Colab Optimized.

This script:
- Loads processed train.csv and validation.csv
- Loads label_map.json for label mapping
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
from typing import List, Dict, Any
import os

# ===== COLAB CONFIGURATION =====
# Set this to True if running in Google Colab
RUNNING_IN_COLAB = os.path.exists('/content')

if RUNNING_IN_COLAB:
    # Colab paths
    BASE_DIR = Path("/content/crisis_model")
    # Optional: Use Google Drive for saving
    USE_DRIVE = True  # Set to False to save locally
    if USE_DRIVE:
        DRIVE_MODEL_DIR = Path("/content/drive/MyDrive/sehatmind/chatbot_backend/crisis_model/model")
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


class CrisisDataset(Dataset):
    """
    Dataset class for crisis detection.
    
    Handles:
    - Text tokenization
    - Binary label encoding (0 = not_crisis, 1 = crisis)
    - Padding and truncation
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128
    ):
        """Initialize the dataset."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_label_map(label_map_path: Path) -> Dict:
    """Load the label mapping from JSON file."""
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    return label_map


def load_processed_data(data_dir: Path, split: str) -> pd.DataFrame:
    """Load processed CSV file."""
    if split == 'validation':
        filepath = data_dir / "validation.csv"
        if not filepath.exists():
            filepath = data_dir / "valid.csv"
    else:
        filepath = data_dir / f"{split}.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data file not found: {filepath}")
    
    df = pd.read_csv(filepath, encoding='utf-8')
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"CSV file must contain 'text' and 'label' columns")
    
    print(f"  [OK] Loaded {split}.csv: {len(df)} rows")
    return df


def compute_metrics(eval_pred):
    """Compute metrics for binary classification."""
    predictions, labels = eval_pred
    
    # Get predicted class (binary classification)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average='binary')
    precision = precision_score(labels, y_pred, average='binary', zero_division=0)
    recall = recall_score(labels, y_pred, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    """Main training function."""
    print("=" * 70)
    print("Crisis Detection Model - Colab Optimized Training")
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
    print(f"  [OK] Loaded label map with {label_map['num_labels']} labels")
    print(f"    Labels: {list(label_map['label_to_id'].keys())}")
    
    # Load processed data
    print("\n[2] Loading processed data...")
    train_df = load_processed_data(PROCESSED_DATA_DIR, 'train')
    validation_df = load_processed_data(PROCESSED_DATA_DIR, 'validation')
    
    # Display label distribution
    print("\n[3] Label distribution in training data:")
    train_label_counts = train_df['label'].value_counts().sort_index()
    for label_id, count in train_label_counts.items():
        label_name = label_map['id_to_label'][str(label_id)]
        pct = count / len(train_df) * 100
        print(f"    {label_name} (label={label_id}): {count} ({pct:.1f}%)")
    
    # Load tokenizer
    print("\n[4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  [OK] Loaded tokenizer: {MODEL_NAME}")
    
    # Create datasets
    print("\n[5] Creating datasets...")
    train_dataset = CrisisDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    validation_dataset = CrisisDataset(
        texts=validation_df['text'].tolist(),
        labels=validation_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    print(f"  [OK] Created training dataset: {len(train_dataset)} examples")
    print(f"  [OK] Created validation dataset: {len(validation_dataset)} examples")
    
    # Load model
    print("\n[6] Loading model...")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    print(f"  [OK] Loaded model: {MODEL_NAME}")
    print(f"    Number of labels: 2")
    print(f"    Problem type: binary classification")
    
    # Configure training arguments
    print("\n[7] Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Larger eval batch
        num_train_epochs=5,  # Similar to emotion model
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Use F1 for binary classification
        greater_is_better=True,
        logging_dir=str(MODEL_OUTPUT_DIR / "logs"),
        logging_steps=100,
        save_total_limit=3,  # Keep more checkpoints
        warmup_steps=300,
        weight_decay=0.01,
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
    print(f"    FP16 (mixed precision): {training_args.fp16}")
    print(f"    Output directory: {training_args.output_dir}")
    
    # Create Trainer
    print("\n[8] Creating Trainer...")
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
    print(f"Number of classes: 2 (binary classification)")
    print(f"\nTraining will run for {training_args.num_train_epochs} epochs...")
    print("This may take 1-3 hours depending on your GPU.\n")
    
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
        print(f"  F1 Score: {eval_results['eval_f1']:.4f}")
        print(f"  Precision: {eval_results['eval_precision']:.4f}")
        print(f"  Recall: {eval_results['eval_recall']:.4f}")
        
        # Save the final model and tokenizer
        print(f"\n[9] Saving model and tokenizer...")
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
            print(f"\n[10] Saving to Google Drive...")
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

