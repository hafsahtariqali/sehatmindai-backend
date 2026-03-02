"""
Training Script for Emotion Detection Model

This script fine-tunes a pretrained emotion detection model on your dataset.
It uses the Hugging Face Transformers library to train the model.

Model: bhadresh-savani/distilbert-base-uncased-emotion
This model detects 6 emotions: sadness, joy, love, anger, fear, surprise

Training Process:
1. Load cleaned data from preprocessing/cleaned/
2. Split into train/validation/test sets
3. Fine-tune the model on training data
4. Evaluate on validation data during training
5. Evaluate on test data after training
6. Save the trained model to models/emotion_classifier/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import torch


# Define directories
BASE_DIR = Path(__file__).parent.parent  # chatbot_backend/
CLEANED_DATA_DIR = BASE_DIR / "preprocessing" / "cleaned"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_MODEL_DIR = MODELS_DIR / "emotion_classifier"


class EmotionDataset(Dataset):
    """
    Custom Dataset class for emotion classification.
    
    This class wraps our data so it can be used with PyTorch and Hugging Face.
    It handles tokenization (converting text to numbers the model can understand).
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings (sentences to classify)
            labels: List of label IDs (numbers representing emotions)
            tokenizer: Tokenizer to convert text to token IDs
            max_length: Maximum length of tokenized sequences (longer texts will be truncated)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        This method is called by PyTorch to get one training example.
        It tokenizes the text and returns it in the format the model expects.
        
        Args:
            idx: Index of the example to retrieve
        
        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'label'
        """
        # Get the text and label at this index
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        # This converts words into numbers the model understands
        # truncation=True: Cut off text if longer than max_length
        # padding='max_length': Add padding if shorter than max_length
        # return_tensors='pt': Return PyTorch tensors
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Return the tokenized text and label
        # Squeeze removes extra dimensions to make tensors the right shape
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_dataset_from_folder(dataset_name, split='train'):
    """
    Load a dataset from the cleaned data folder.
    
    Looks for files like: {dataset_name}_{split}.csv
    Example: empathetic_dialogue_train.csv
    
    Args:
        dataset_name: Name of the dataset folder
        split: Which split to load ('train', 'val', or 'test')
    
    Returns:
        DataFrame with 'text' and 'label' columns, or None if not found
    """
    # Construct the expected filename
    filename = f"{dataset_name}_{split}.csv"
    filepath = CLEANED_DATA_DIR / dataset_name / filename
    
    if not filepath.exists():
        print(f"  Warning: {filename} not found. Skipping...")
        return None
    
    # Load the CSV file
    df = pd.read_csv(filepath)
    
    # Verify required columns exist
    if 'text' not in df.columns:
        print(f"  Error: 'text' column not found in {filename}")
        return None
    
    # Remove any rows with missing text or labels
    df = df.dropna(subset=['text'])
    
    # If labels are missing, we can't use this for training
    if 'label' not in df.columns:
        print(f"  Warning: 'label' column not found in {filename}. This split cannot be used for training.")
        return None
    
    df = df.dropna(subset=['label'])
    
    print(f"  Loaded {len(df)} examples from {filename}")
    return df


def combine_datasets(dataset_names, split='train'):
    """
    Combine multiple datasets into one.
    
    This allows you to train on multiple datasets at once.
    
    Args:
        dataset_names: List of dataset folder names to combine
        split: Which split to load ('train', 'val', or 'test')
    
    Returns:
        Combined DataFrame with all examples
    """
    all_dataframes = []
    
    for dataset_name in dataset_names:
        df = load_dataset_from_folder(dataset_name, split)
        if df is not None and not df.empty:
            all_dataframes.append(df)
    
    if not all_dataframes:
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"  Combined {len(combined_df)} total examples from {len(all_dataframes)} dataset(s)")
    
    return combined_df


def prepare_label_encoding(train_df):
    """
    Create a mapping between emotion labels and numbers.
    
    The model needs numbers (0, 1, 2, etc.) but our data has text labels
    (like 'joy', 'sadness', etc.). This function creates a mapping.
    
    Args:
        train_df: Training DataFrame with 'label' column
    
    Returns:
        label2id: Dictionary mapping label text to ID number
        id2label: Dictionary mapping ID number to label text
        num_labels: Total number of unique labels
    """
    # Get unique labels from training data
    unique_labels = sorted(train_df['label'].unique().tolist())
    
    # Create mapping: label text -> number
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Create reverse mapping: number -> label text
    id2label = {idx: label for label, idx in label2id.items()}
    
    num_labels = len(unique_labels)
    
    print(f"\nLabel Encoding:")
    print(f"  Found {num_labels} unique emotions: {unique_labels}")
    print(f"  Label mapping: {label2id}")
    
    return label2id, id2label, num_labels


def compute_metrics(eval_pred):
    """
    Calculate accuracy and F1 score for evaluation.
    
    This function is called by the Trainer during evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels) from the model
    
    Returns:
        Dictionary with 'accuracy' and 'f1' scores
    """
    # Unpack predictions and true labels
    predictions, labels = eval_pred
    
    # Get the predicted class (the emotion with highest probability)
    # predictions shape: (num_examples, num_classes)
    # We want the class index with highest score for each example
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy: percentage of correct predictions
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate F1 score: harmonic mean of precision and recall
    # average='weighted': Calculate F1 for each class, then average weighted by support
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def train_emotion_model(
    dataset_names=None,
    model_name="bhadresh-savani/distilbert-base-uncased-emotion",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_length=128
):
    """
    Main training function.
    
    This function orchestrates the entire training process:
    1. Load data
    2. Prepare labels
    3. Initialize model and tokenizer
    4. Create datasets
    5. Train the model
    6. Evaluate on test set
    7. Save the model
    
    Args:
        dataset_names: List of dataset names to use. If None, uses all available datasets.
        model_name: Name of the pretrained model to fine-tune
        num_epochs: Number of training epochs (how many times to go through the data)
        batch_size: Number of examples to process at once
        learning_rate: How fast the model learns (lower = slower but more stable)
        max_length: Maximum length of input text in tokens
    """
    
    print("=" * 70)
    print("Emotion Detection Model Training")
    print("=" * 70)
    
    # Step 1: Load datasets
    print("\n[Step 1] Loading datasets...")
    
    # If no specific datasets provided, find all available ones
    if dataset_names is None:
        available_datasets = [d.name for d in CLEANED_DATA_DIR.iterdir() 
                            if d.is_dir() and not d.name.startswith('.')]
        dataset_names = available_datasets
        print(f"  No specific datasets specified. Using all available: {dataset_names}")
    else:
        print(f"  Using specified datasets: {dataset_names}")
    
    # Load training data
    print("\n  Loading training data...")
    train_df = combine_datasets(dataset_names, split='train')
    
    if train_df.empty:
        print("  ERROR: No training data found!")
        print("  Please run preprocessing/load_datasets.py first to prepare your data.")
        return
    
    # Load validation data
    print("\n  Loading validation data...")
    val_df = combine_datasets(dataset_names, split='val')
    
    if val_df.empty:
        print("  WARNING: No validation data found!")
        print("  Training will proceed without validation, which is not recommended.")
        val_df = None
    
    # Load test data (only for final evaluation, not for training)
    print("\n  Loading test data...")
    test_df = combine_datasets(dataset_names, split='test')
    
    if test_df.empty:
        print("  WARNING: No test data found!")
        print("  Will skip final test evaluation.")
        test_df = None
    
    # Step 2: Prepare label encoding
    print("\n[Step 2] Preparing label encoding...")
    label2id, id2label, num_labels = prepare_label_encoding(train_df)
    
    # Convert text labels to numeric IDs for training
    train_df['label_id'] = train_df['label'].map(label2id)
    if val_df is not None:
        val_df['label_id'] = val_df['label'].map(label2id)
    if test_df is not None:
        test_df['label_id'] = test_df['label'].map(label2id)
    
    # Step 3: Load tokenizer and model
    print(f"\n[Step 3] Loading model and tokenizer...")
    print(f"  Model: {model_name}")
    
    try:
        # Load the tokenizer (converts text to numbers)
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("  ✓ Tokenizer loaded")
        
        # Load the pretrained model
        # num_labels tells the model how many emotion classes to predict
        print("  Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        )
        print("  ✓ Model loaded")
        
    except Exception as e:
        print(f"  ERROR: Failed to load model: {e}")
        print("  Make sure you have internet connection for first-time download.")
        return
    
    # Step 4: Create PyTorch datasets
    print("\n[Step 4] Creating training datasets...")
    
    # Create dataset objects that PyTorch can use
    train_dataset = EmotionDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    print(f"  ✓ Training dataset created: {len(train_dataset)} examples")
    
    if val_df is not None:
        val_dataset = EmotionDataset(
            texts=val_df['text'].tolist(),
            labels=val_df['label_id'].tolist(),
            tokenizer=tokenizer,
            max_length=max_length
        )
        print(f"  ✓ Validation dataset created: {len(val_dataset)} examples")
    else:
        val_dataset = None
    
    if test_df is not None:
        test_dataset = EmotionDataset(
            texts=test_df['text'].tolist(),
            labels=test_df['label_id'].tolist(),
            tokenizer=tokenizer,
            max_length=max_length
        )
        print(f"  ✓ Test dataset created: {len(test_dataset)} examples")
    else:
        test_dataset = None
    
    # Step 5: Set up training arguments
    print("\n[Step 5] Configuring training parameters...")
    
    # Create output directory for training artifacts
    training_output_dir = OUTPUT_MODEL_DIR / "training_output"
    training_output_dir.mkdir(parents=True, exist_ok=True)
    
    # TrainingArguments controls all aspects of training
    training_args = TrainingArguments(
        # Output directory for checkpoints and logs
        output_dir=str(training_output_dir),
        
        # Number of training epochs (full passes through the data)
        num_train_epochs=num_epochs,
        
        # Batch size: how many examples to process at once
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        
        # Learning rate: how fast to update model weights
        learning_rate=learning_rate,
        
        # Weight decay: regularization to prevent overfitting
        weight_decay=0.01,
        
        # Evaluation settings
        evaluation_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="epoch",
        
        # Save the best model based on validation loss
        load_best_model_at_end=True if val_dataset is not None else False,
        metric_for_best_model="f1" if val_dataset is not None else None,
        
        # Logging
        logging_dir=str(training_output_dir / "logs"),
        logging_steps=100,  # Log every 100 steps
        
        # Save total limit: keep only the best and last checkpoint
        save_total_limit=2,
        
        # Other useful settings
        warmup_steps=500,  # Gradually increase learning rate at start
        report_to="none",  # Disable wandb/tensorboard (can enable if needed)
    )
    
    print(f"  Training epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Output directory: {training_output_dir}")
    
    # Step 6: Create Trainer
    print("\n[Step 6] Setting up Trainer...")
    
    # Callbacks for training
    callbacks = []
    if val_dataset is not None:
        # Early stopping: stop training if validation performance stops improving
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
    
    # Trainer handles the training loop
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    print("  ✓ Trainer configured")
    
    # Step 7: Train the model
    print("\n[Step 7] Starting training...")
    print("  This may take a while depending on your data size and hardware.")
    print("  Training progress will be shown below:\n")
    
    try:
        # This is where the actual training happens!
        # The model will learn to predict emotions from your training data
        train_result = trainer.train()
        
        print("\n  ✓ Training completed!")
        print(f"  Training loss: {train_result.training_loss:.4f}")
        
    except Exception as e:
        print(f"\n  ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 8: Evaluate on validation set (if available)
    if val_dataset is not None:
        print("\n[Step 8] Evaluating on validation set...")
        val_metrics = trainer.evaluate(eval_dataset=val_dataset)
        
        print(f"  Validation Accuracy: {val_metrics['eval_accuracy']:.4f} ({val_metrics['eval_accuracy']*100:.2f}%)")
        print(f"  Validation F1 Score: {val_metrics['eval_f1']:.4f}")
    
    # Step 9: Final evaluation on test set
    if test_dataset is not None:
        print("\n[Step 9] Final evaluation on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        
        print(f"\n  {'='*70}")
        print(f"  FINAL TEST RESULTS:")
        print(f"  {'='*70}")
        print(f"  Test Accuracy: {test_metrics['eval_accuracy']:.4f} ({test_metrics['eval_accuracy']*100:.2f}%)")
        print(f"  Test F1 Score: {test_metrics['eval_f1']:.4f}")
        
        # Detailed classification report
        print(f"\n  Detailed Classification Report:")
        print(f"  {'-'*70}")
        
        # Get predictions on test set
        test_predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(test_predictions.predictions, axis=1)
        y_true = test_predictions.label_ids
        
        # Print per-class metrics
        report = classification_report(
            y_true, y_pred,
            target_names=[id2label[i] for i in range(num_labels)],
            digits=4
        )
        print(report)
    
    # Step 10: Save the trained model
    print("\n[Step 10] Saving trained model...")
    
    try:
        # Save the model and tokenizer to the output directory
        OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_pretrained(str(OUTPUT_MODEL_DIR))
        print(f"  ✓ Model saved to: {OUTPUT_MODEL_DIR}")
        
        # Save tokenizer
        tokenizer.save_pretrained(str(OUTPUT_MODEL_DIR))
        print(f"  ✓ Tokenizer saved")
        
        # Save label mappings
        import json
        with open(OUTPUT_MODEL_DIR / "label_mappings.json", "w") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
        print(f"  ✓ Label mappings saved")
        
    except Exception as e:
        print(f"  ERROR saving model: {e}")
        return
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print(f"Model saved to: {OUTPUT_MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    """
    This block runs when the script is executed directly.
    """
    
    # You can customize these parameters:
    
    # Which datasets to use for training
    # Set to None to use all available datasets
    # Or specify a list: ['empathetic_dialogue', 'dailydialog']
    DATASET_NAMES = None
    
    # Training hyperparameters
    NUM_EPOCHS = 3  # Number of times to go through the training data
    BATCH_SIZE = 16  # Number of examples per batch
    LEARNING_RATE = 2e-5  # How fast the model learns
    MAX_LENGTH = 128  # Maximum text length in tokens
    
    # Run training
    train_emotion_model(
        dataset_names=DATASET_NAMES,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH
    )

