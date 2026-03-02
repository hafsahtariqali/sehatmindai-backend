"""
Evaluation script for crisis detection model.

This script:
- Loads trained model and tokenizer
- Loads processed test.csv
- Computes comprehensive evaluation metrics:
  - Precision
  - Recall (most important for crisis detection)
  - F1 score
  - Confusion matrix
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
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import List, Dict


# Define paths
BASE_DIR = Path(__file__).parent  # crisis_model/
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


def load_model_and_tokenizer(model_dir: Path):
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
        
        # Check for latest checkpoint or look for trainer_state.json
        for checkpoint_dir in reversed(checkpoint_dirs):
            trainer_state_path = checkpoint_dir / "trainer_state.json"
            if trainer_state_path.exists():
                try:
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                        if 'best_global_step' in trainer_state:
                            best_step = trainer_state['best_global_step']
                            current_step = int(checkpoint_dir.name.split('-')[-1])
                            if best_step == current_step:
                                best_checkpoint = checkpoint_dir
                                print(f"  [OK] Found best checkpoint: {checkpoint_dir.name}")
                                break
                except:
                    pass
        
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
        Array of predictions (class IDs)
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
            logits = outputs.logits
            
            # Get predicted class (argmax for binary classification)
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        all_predictions.append(predictions)
    
    return np.concatenate(all_predictions)


def print_confusion_matrix(cm: np.ndarray, label_map: Dict):
    """
    Print confusion matrix in a clear format.
    
    Args:
        cm: Confusion matrix array
        label_map: Label mapping dictionary
    """
    id_to_label = label_map["id_to_label"]
    
    print("\n[CONFUSION MATRIX]")
    print("-" * 70)
    print(f"{'':<15} {'Predicted: Not Crisis':<25} {'Predicted: Crisis':<25}")
    print("-" * 70)
    
    # Row 1: True = Not Crisis (label 0)
    true_label_0 = id_to_label.get("0", "not_crisis")
    print(f"True: {true_label_0:<10} {cm[0, 0]:<25} {cm[0, 1]:<25}")
    print(f"                    (True Negative)          (False Positive)")
    
    # Row 2: True = Crisis (label 1)
    true_label_1 = id_to_label.get("1", "crisis")
    print(f"True: {true_label_1:<10} {cm[1, 0]:<25} {cm[1, 1]:<25}")
    print(f"                    (False Negative)        (True Positive)")
    print("-" * 70)
    
    # Extract values
    tn = cm[0, 0]  # True Negative
    fp = cm[0, 1]  # False Positive
    fn = cm[1, 0]  # False Negative (MISSED CRISIS - CRITICAL!)
    tp = cm[1, 1]  # True Positive
    
    print(f"\n  True Negatives (TN):  {tn} (correctly identified as not crisis)")
    print(f"  False Positives (FP): {fp} (incorrectly flagged as crisis)")
    print(f"  False Negatives (FN): {fn} (MISSED CRISIS - CRITICAL ERROR)")
    print(f"  True Positives (TP):  {tp} (correctly identified crisis)")
    
    # Highlight false negatives as critical
    if fn > 0:
        print(f"\n  [WARN] {fn} crisis cases were MISSED (False Negatives).")
        print(f"         This is critical - all crisis cases must be detected!")
        print(f"         Consider lowering the decision threshold or improving the model.")


def main():
    """Main evaluation function."""
    print("=" * 70)
    print("Crisis Detection Model - Evaluation")
    print("=" * 70)
    
    # Load label map
    print("\n[1] Loading label mapping...")
    label_map = load_label_map(LABEL_MAP_PATH)
    num_labels = label_map['num_labels']
    id_to_label = label_map["id_to_label"]
    print(f"  [OK] Loaded label map with {num_labels} labels")
    print(f"    Labels: {list(label_map['label_to_id'].keys())}")
    
    # Load model and tokenizer
    print("\n[2] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR)
    
    # Load test data
    print("\n[3] Loading test data...")
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")
    
    test_df = pd.read_csv(TEST_DATA_PATH, encoding='utf-8')
    print(f"  [OK] Loaded {len(test_df)} test examples")
    
    # Validate columns
    if 'text' not in test_df.columns or 'label' not in test_df.columns:
        raise ValueError("Test CSV must contain 'text' and 'label' columns")
    
    # Get true labels
    y_true = test_df['label'].values.astype(int)
    
    # Show label distribution
    print("\n[4] Test set label distribution:")
    unique, counts = np.unique(y_true, return_counts=True)
    for label_id, count in zip(unique, counts):
        label_name = id_to_label[str(label_id)]
        pct = count / len(y_true) * 100
        print(f"    {label_name} (label={label_id}): {count} ({pct:.1f}%)")
    
    # Make predictions
    print("\n[5] Making predictions on test set...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    texts = test_df['text'].tolist()
    y_pred = predict_batch(model, tokenizer, texts, device=device, batch_size=32)
    
    print(f"  [OK] Predictions completed")
    
    # Show predicted distribution
    print("\n[6] Predicted label distribution:")
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    for label_id, count in zip(unique_pred, counts_pred):
        label_name = id_to_label[str(label_id)]
        pct = count / len(y_pred) * 100
        print(f"    {label_name} (label={label_id}): {count} ({pct:.1f}%)")
    
    # Compute metrics
    print("\n[7] Computing evaluation metrics...")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 for binary classification
    # For crisis detection, we focus on the 'crisis' class (label=1)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print("\n[OVERALL METRICS]")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%) ⚠️  MOST IMPORTANT")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print("\n[PER-CLASS METRICS]")
    print("-" * 70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 70)
    
    for label_id in [0, 1]:
        label_name = id_to_label[str(label_id)]
        idx = label_id
        print(f"{label_name:<15} {precision_per_class[idx]:<12.4f} {recall_per_class[idx]:<12.4f} "
              f"{f1_per_class[idx]:<12.4f}")
    
    print("-" * 70)
    
    # Confusion matrix
    print_confusion_matrix(cm, label_map)
    
    # Detailed analysis
    print("\n[CRITICAL METRICS FOR CRISIS DETECTION]")
    print("-" * 70)
    
    # Extract confusion matrix values
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    
    # Calculate critical rates
    if tp + fn > 0:
        sensitivity = tp / (tp + fn)  # Same as recall
        false_negative_rate = fn / (tp + fn)
    else:
        sensitivity = 0.0
        false_negative_rate = 0.0
    
    if tn + fp > 0:
        specificity = tn / (tn + fp)
        false_positive_rate = fp / (tn + fp)
    else:
        specificity = 0.0
        false_positive_rate = 0.0
    
    print(f"  Sensitivity (Recall for Crisis): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"    → Percentage of actual crises correctly detected")
    print(f"    → Goal: > 95% (critical for safety)")
    
    print(f"\n  False Negative Rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
    print(f"    → Percentage of crises MISSED")
    print(f"    → Goal: < 5% (must be minimized)")
    
    if fn > 0:
        print(f"\n  ⚠️  WARNING: {fn} crisis case(s) were MISSED!")
        print(f"     This is critical for safety. Consider:")
        print(f"     1. Lowering the decision threshold")
        print(f"     2. Adding more crisis training examples")
        print(f"     3. Improving model training")
    
    print(f"\n  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"    → Percentage of non-crisis correctly identified")
    
    print(f"\n  False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    print(f"    → Percentage of non-crisis incorrectly flagged as crisis")
    
    # Summary
    print("\n[SUMMARY]")
    print("-" * 70)
    print(f"  Total test examples: {len(test_df)}")
    print(f"  True crises: {(y_true == 1).sum()}")
    print(f"  Predicted crises: {(y_pred == 1).sum()}")
    print(f"  Correctly detected crises: {tp}")
    print(f"  Missed crises (CRITICAL): {fn}")
    print(f"  False alarms: {fp}")
    
    if recall >= 0.95:
        print(f"\n  ✓ [OK] Recall ({recall:.2%}) meets safety threshold (≥95%)")
    else:
        print(f"\n  ✗ [WARN] Recall ({recall:.2%}) is below safety threshold (95%)")
        print(f"     Model needs improvement before deployment!")
    
    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)
    
    # Return metrics for potential further use
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'false_negatives': fn,
        'false_positives': fp,
        'true_positives': tp,
        'true_negatives': tn
    }


if __name__ == "__main__":
    main()
