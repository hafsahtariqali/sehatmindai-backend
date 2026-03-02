"""
Evaluation script for emotion classification model.

This script:
- Loads trained model and tokenizer
- Loads processed test.csv
- Computes comprehensive evaluation metrics:
  - Accuracy
  - Macro F1 score
  - Micro F1 score
  - Per-emotion precision and recall
- Prints results clearly
"""

import pandas as pd
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from typing import List, Dict, Tuple
import ast


# Define paths
BASE_DIR = Path(__file__).parent  # emotion_model/
MODEL_DIR = BASE_DIR / "model"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
MAX_LENGTH = 128


def load_label_map(label_map_path: Path) -> Dict:
    """Load the label mapping from JSON file."""
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    return label_map


def load_model_and_tokenizer(model_dir: Path, label_map: Dict):
    """Load the trained model and tokenizer."""
    print("[1] Loading model and tokenizer...")
    
    # Find best checkpoint (same logic as predict.py)
    checkpoint_dirs = sorted(
        [d for d in model_dir.iterdir() 
         if d.is_dir() and d.name.startswith('checkpoint')],
        key=lambda x: int(x.name.split('-')[-1]) if x.name.split('-')[-1].isdigit() else 0
    )
    
    if checkpoint_dirs:
        # Try to find best checkpoint
        best_checkpoint = None
        
        # Check for checkpoint-10707 or latest
        target_checkpoint = model_dir / "checkpoint-10707"
        if target_checkpoint.exists() and target_checkpoint.is_dir():
            model_files = list(target_checkpoint.glob("*.safetensors")) + list(target_checkpoint.glob("*.bin"))
            if model_files:
                best_checkpoint = target_checkpoint
                print(f"  [OK] Using checkpoint-10707")
        
        if best_checkpoint is None:
            # Use latest checkpoint
            best_checkpoint = checkpoint_dirs[-1]
            print(f"  [OK] Using latest checkpoint: {best_checkpoint.name}")
        
        model_path = best_checkpoint
    else:
        model_path = model_dir
        print(f"  [OK] Using root model directory")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    print(f"  [OK] Model loaded")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print(f"  [OK] Tokenizer loaded")
    
    return model, tokenizer


def labels_to_multihot(label_string: str, label_map: Dict, num_labels: int) -> np.ndarray:
    """
    Convert pipe-separated labels to multi-hot vector.
    
    Args:
        label_string: Pipe-separated label string (e.g., "joy|sadness")
        label_map: Label mapping dictionary
        num_labels: Number of labels
    
    Returns:
        Multi-hot vector as numpy array
    """
    multihot = np.zeros(num_labels, dtype=np.float32)
    
    if pd.isna(label_string) or label_string == '':
        return multihot
    
    # Split by pipe separator
    labels = str(label_string).split('|')
    
    # Set corresponding indices to 1
    for label in labels:
        label = label.strip()
        if label in label_map["label_to_id"]:
            label_id = label_map["label_to_id"][label]
            multihot[label_id] = 1.0
    
    return multihot


def predict_batch(model, tokenizer, texts: List[str], device: str = 'cpu', batch_size: int = 32):
    """
    Make predictions on a batch of texts.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        texts: List of text strings
        device: Device to run on
        batch_size: Batch size for prediction
    
    Returns:
        Array of predictions (logits)
    """
    model.to(device)
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
        
        all_predictions.append(logits)
    
    return np.vstack(all_predictions)


def compute_metrics_multi_label(y_true: np.ndarray, y_pred: np.ndarray, label_map: Dict) -> Dict:
    """
    Compute comprehensive metrics for multi-label classification.
    
    Args:
        y_true: True labels (multi-hot vectors)
        y_pred: Predicted labels (multi-hot vectors)
        label_map: Label mapping dictionary
    
    Returns:
        Dictionary with all metrics
    """
    # Per-sample accuracy (exact match)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro-averaged metrics (average across all labels)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Micro-averaged metrics (global metrics)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Weighted F1 (weighted by support)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-emotion metrics
    per_emotion_metrics = {}
    id_to_label = label_map["id_to_label"]
    
    for label_id in range(len(id_to_label)):
        emotion_name = id_to_label[str(label_id)]
        
        # Get true and predicted labels for this emotion
        y_true_emotion = y_true[:, label_id]
        y_pred_emotion = y_pred[:, label_id]
        
        # Calculate metrics for this emotion
        precision = precision_score(y_true_emotion, y_pred_emotion, zero_division=0)
        recall = recall_score(y_true_emotion, y_pred_emotion, zero_division=0)
        f1 = f1_score(y_true_emotion, y_pred_emotion, zero_division=0)
        
        # Support (number of true positives)
        support = int(y_true_emotion.sum())
        
        per_emotion_metrics[emotion_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'per_emotion': per_emotion_metrics
    }


def main():
    """Main evaluation function."""
    print("=" * 70)
    print("Emotion Classification Model - Evaluation")
    print("=" * 70)
    
    # Load label map
    print("\n[1] Loading label mapping...")
    label_map = load_label_map(LABEL_MAP_PATH)
    num_labels = label_map['num_labels']
    print(f"  [OK] Loaded label map with {num_labels} emotions")
    print(f"    Emotions: {list(label_map['label_to_id'].keys())}")
    
    # Load model and tokenizer
    print("\n[2] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR, label_map)
    
    # Load test data
    print("\n[3] Loading test data...")
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")
    
    test_df = pd.read_csv(TEST_DATA_PATH, encoding='utf-8')
    print(f"  [OK] Loaded {len(test_df)} test examples")
    
    # Validate columns
    if 'text' not in test_df.columns or 'labels' not in test_df.columns:
        raise ValueError("Test CSV must contain 'text' and 'labels' columns")
    
    # Convert labels to multi-hot vectors
    print("\n[4] Converting labels to multi-hot vectors...")
    y_true_list = []
    texts = []
    
    for idx, row in test_df.iterrows():
        text = str(row['text']).strip()
        label_string = str(row['labels']) if pd.notna(row['labels']) else ""
        
        multihot = labels_to_multihot(label_string, label_map, num_labels)
        y_true_list.append(multihot)
        texts.append(text)
    
    y_true = np.array(y_true_list)
    print(f"  [OK] Converted {len(y_true)} examples")
    print(f"    Active labels per example: {y_true.sum(axis=1).mean():.2f} (average)")
    
    # Make predictions
    print("\n[5] Making predictions on test set...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    # Get logits from model
    logits = predict_batch(model, tokenizer, texts, device=device, batch_size=32)
    
    # Apply sigmoid to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(torch.Tensor(logits)).numpy()
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = np.zeros(probabilities.shape, dtype=np.float32)
    y_pred[probabilities >= 0.5] = 1.0
    
    print(f"  [OK] Predictions completed")
    print(f"    Predicted labels per example: {y_pred.sum(axis=1).mean():.2f} (average)")
    
    # Compute metrics
    print("\n[6] Computing evaluation metrics...")
    metrics = compute_metrics_multi_label(y_true, y_pred, label_map)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print("\n[OVERALL METRICS]")
    print(f"  Accuracy (exact match):     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1 Macro (avg across labels): {metrics['f1_macro']:.4f}")
    print(f"  F1 Micro (global):         {metrics['f1_micro']:.4f}")
    print(f"  F1 Weighted:                {metrics['f1_weighted']:.4f}")
    print(f"  Precision Macro:           {metrics['precision_macro']:.4f}")
    print(f"  Recall Macro:              {metrics['recall_macro']:.4f}")
    print(f"  Precision Micro:           {metrics['precision_micro']:.4f}")
    print(f"  Recall Micro:              {metrics['recall_micro']:.4f}")
    
    print("\n[PER-EMOTION METRICS]")
    print("-" * 70)
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Support':<10}")
    print("-" * 70)
    
    per_emotion = metrics['per_emotion']
    for emotion in sorted(per_emotion.keys()):
        emotion_metrics = per_emotion[emotion]
        print(f"{emotion:<12} {emotion_metrics['precision']:<12.4f} {emotion_metrics['recall']:<12.4f} "
              f"{emotion_metrics['f1']:<12.4f} {emotion_metrics['support']:<10}")
    
    print("-" * 70)
    
    # Summary statistics
    print("\n[SUMMARY]")
    print(f"  Total test examples: {len(test_df)}")
    print(f"  Number of emotion classes: {num_labels}")
    print(f"  Average labels per example (true): {y_true.sum(axis=1).mean():.2f}")
    print(f"  Average labels per example (predicted): {y_pred.sum(axis=1).mean():.2f}")
    
    # Find best and worst performing emotions
    emotion_f1_scores = {emotion: metrics['per_emotion'][emotion]['f1'] 
                         for emotion in per_emotion.keys()}
    best_emotion = max(emotion_f1_scores, key=emotion_f1_scores.get)
    worst_emotion = min(emotion_f1_scores, key=emotion_f1_scores.get)
    
    print(f"\n  Best performing emotion: {best_emotion} (F1: {emotion_f1_scores[best_emotion]:.4f})")
    print(f"  Worst performing emotion: {worst_emotion} (F1: {emotion_f1_scores[worst_emotion]:.4f})")
    
    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

