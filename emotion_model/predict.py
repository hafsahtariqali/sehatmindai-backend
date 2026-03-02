"""
Prediction script for emotion classification model.

This script:
- Loads trained model and tokenizer from emotion_model/model/
- Accepts user input text
- Predicts emotion probabilities
- Returns top emotions with confidence scores
"""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import numpy as np


# Define paths
BASE_DIR = Path(__file__).parent  # emotion_model/
MODEL_DIR = BASE_DIR / "model"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
MAX_LENGTH = 128


class EmotionPredictor:
    """
    Emotion prediction class that loads and uses the trained model.
    """
    
    def __init__(self, model_dir: Path = MODEL_DIR):
        """
        Initialize the emotion predictor.
        
        Args:
            model_dir: Directory containing the trained model
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.id_to_label = None
        self.num_labels = None
        
        # Load components
        self._load_model()
        self._load_tokenizer()
        self._load_label_map()
    
    def _load_model(self):
        """Load the trained model."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                f"Please ensure the trained model is in this location."
            )
        
        # IMPORTANT: Always check for checkpoint directories first
        # The root model/ directory may contain an INITIAL (untrained) model save
        # The actual trained weights are in checkpoint-XXXXX folders
        print(f"[SEARCH] Searching for checkpoints in: {self.model_dir}")
        
        checkpoint_dirs = sorted(
            [d for d in self.model_dir.iterdir() 
             if d.is_dir() and d.name.startswith('checkpoint')],
            key=lambda x: int(x.name.split('-')[-1]) if x.name.split('-')[-1].isdigit() else 0
        )
        
        print(f"   Found {len(checkpoint_dirs)} checkpoint directory(ies): {[d.name for d in checkpoint_dirs]}")
        
        if checkpoint_dirs:
            # ALWAYS use checkpoint directories if they exist - they contain the trained weights
            best_checkpoint = None
            
            # First, try to find checkpoint-10707 specifically (known best model from training)
            target_checkpoint = self.model_dir / "checkpoint-10707"
            if target_checkpoint.exists() and target_checkpoint.is_dir():
                model_files_in_checkpoint = list(target_checkpoint.glob("*.safetensors")) + list(target_checkpoint.glob("*.bin"))
                if model_files_in_checkpoint:
                    best_checkpoint = target_checkpoint
                    file_size_mb = model_files_in_checkpoint[0].stat().st_size / (1024 * 1024)
                    print(f"[OK] Found checkpoint-10707 with model file ({file_size_mb:.1f} MB)")
                    print(f"  This is the final trained model from epoch 3")
                else:
                    print(f"  [WARN] checkpoint-10707 exists but has no model files!")
            
            # If checkpoint-10707 not found, try to find the best model via trainer_state.json
            if best_checkpoint is None:
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
                                        print(f"[OK] Found best model checkpoint: {checkpoint_dir.name}")
                                        print(f"  Best step: {best_step}, Best metric: {trainer_state.get('best_metric', 'N/A')}")
                                        break
                        except Exception as e:
                            print(f"  [WARN] Could not read trainer_state.json from {checkpoint_dir.name}: {e}")
                            continue
            
            # If still no best checkpoint found, use the latest (highest number)
            if best_checkpoint is None:
                best_checkpoint = checkpoint_dirs[-1]
                print(f"[OK] Using latest checkpoint: {best_checkpoint.name} (step {best_checkpoint.name.split('-')[-1]})")
            
            # Verify the selected checkpoint has model files
            checkpoint_model_files = list(best_checkpoint.glob("*.safetensors")) + list(best_checkpoint.glob("*.bin"))
            if not checkpoint_model_files:
                print(f"\n[ERROR] Selected checkpoint '{best_checkpoint.name}' has no model files!")
                print(f"   Expected: {best_checkpoint}/model.safetensors or {best_checkpoint}/pytorch_model.bin")
                print(f"   This checkpoint folder is empty or incomplete.")
                print(f"\n   Please verify:")
                print(f"   1. Did you download the entire 'model' folder from Colab?")
                print(f"   2. Are the checkpoint-XXXXX subdirectories included?")
                print(f"   3. Does {best_checkpoint}/ contain model.safetensors?")
                raise FileNotFoundError(
                    f"Checkpoint '{best_checkpoint.name}' has no model files. "
                    f"Please re-download the model folder from Colab, ensuring all checkpoint subdirectories are included."
                )
            
            model_path = best_checkpoint
            print(f"[PATH] Model path selected: {model_path}")
        
        else:
            # No checkpoint directories found - fall back to root directory
            # WARNING: If you trained in Colab, you should have checkpoint folders!
            # The root model.safetensors might be from initialization (untrained)
            print(f"[WARN] WARNING: No checkpoint directories found!")
            print(f"   If you trained the model, checkpoint folders should exist.")
            print(f"   Falling back to root directory model files...")
            
            model_files = list(self.model_dir.glob("*.bin")) + list(self.model_dir.glob("*.safetensors"))
            config_file = self.model_dir / "config.json"
            
            if config_file.exists() and model_files:
                model_path = self.model_dir
                print(f"   Using root directory model (this may be UNTRAINED if checkpoints exist!)")
            else:
                raise FileNotFoundError(
                    f"No model files found in {self.model_dir}\n"
                    f"Expected: config.json and pytorch_model.bin or model.safetensors\n"
                    f"Or checkpoint directories (checkpoint-10707, etc.)\n\n"
                    f"If you downloaded the model from Colab, make sure you downloaded\n"
                    f"the entire 'model' folder including all checkpoint-XXXXX subdirectories."
                )
        
        print(f"Loading model from {model_path}...")
        
        # Verify model files exist - prioritize .safetensors over .bin, exclude training_args.bin
        safetensors_files = list(model_path.glob("*.safetensors"))
        bin_files = [f for f in model_path.glob("*.bin") if f.name != "training_args.bin"]
        
        # Prioritize safetensors files (newer format)
        if safetensors_files:
            model_files = safetensors_files
        elif bin_files:
            model_files = bin_files
        else:
            model_files = []
        
        if not model_files:
            raise FileNotFoundError(
                f"No model weights found in {model_path}\n"
                f"Expected: pytorch_model.bin or model.safetensors (not training_args.bin)"
            )
        
        # Show which model file is being loaded
        model_file = model_files[0]
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  Loading weights from: {model_file.name} ({file_size_mb:.1f} MB)")
        
        # Verify file size is reasonable (should be ~260MB for DistilBERT)
        if file_size_mb < 100:
            print(f"  [WARN] Warning: Model file is smaller than expected (<100MB)")
            print(f"     A trained DistilBERT model should be ~260MB")
            print(f"     This might be an incomplete or corrupted file")
        elif file_size_mb > 50:
            print(f"  [OK] Model file size looks correct")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        )
        self.model.eval()  # Set to evaluation mode
        
        # Verify model configuration
        print(f"  Model type: {type(self.model).__name__}")
        if hasattr(self.model.config, 'num_labels'):
            print(f"  Number of labels: {self.model.config.num_labels}")
        if hasattr(self.model.config, 'problem_type'):
            print(f"  Problem type: {self.model.config.problem_type}")
        
        print("[OK] Model loaded successfully")
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        print(f"Loading tokenizer from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        print("[OK] Tokenizer loaded successfully")
    
    def _load_label_map(self):
        """Load the label mapping."""
        if not LABEL_MAP_PATH.exists():
            # Try alternative location
            alt_path = BASE_DIR / "label_map.json"
            if alt_path.exists():
                label_map_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Label map not found at {LABEL_MAP_PATH} or {alt_path}"
                )
        else:
            label_map_path = LABEL_MAP_PATH
        
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        # Create reverse mapping (ID to label)
        self.id_to_label = self.label_map["id_to_label"]
        self.num_labels = self.label_map["num_labels"]
        
        print(f"[OK] Label map loaded: {self.num_labels} emotions")
        print(f"  Emotions: {list(self.label_map['label_to_id'].keys())}")
    
    def predict(self, text: str, top_k: int = 3, threshold: float = 0.05) -> Dict:
        """
        Predict emotions for given text.
        
        Args:
            text: Input text to classify
            top_k: Number of top emotions to return
            threshold: Minimum confidence threshold (0.0 to 1.0)
        
        Returns:
            Dictionary containing:
            - emotions: List of predicted emotions with confidence scores
            - top_emotion: The emotion with highest confidence
            - top_confidence: Confidence score of top emotion
            - all_probabilities: Dictionary of all emotion probabilities
        """
        if not text or not text.strip():
            return {
                "emotions": [],
                "top_emotion": None,
                "top_confidence": 0.0,
                "all_probabilities": {}
            }
        
        # Tokenize input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Apply sigmoid to get probabilities (multi-label classification)
        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(logits).cpu().numpy()[0]
        
        # Debug: Check raw logits and probabilities
        raw_logits = logits.cpu().numpy()[0]
        max_logit = np.max(raw_logits)
        min_logit = np.min(raw_logits)
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities)
        
        # Debug output if probabilities are suspiciously low
        # This helps diagnose if model is untrained or checkpoint is wrong
        if max_prob < 0.1:  # Less than 10%
            print(f"\n[DEBUG] Model Output Analysis:")
            print(f"  Raw logits range: [{min_logit:.4f}, {max_logit:.4f}]")
            print(f"  Probabilities range: [{min_prob:.6f}, {max_prob:.6f}]")
            print(f"  Raw logits values: {[f'{l:.2f}' for l in raw_logits]}")
            print(f"  Probabilities values: {[f'{p:.4f}' for p in probabilities]}")
            
            # If logits are all very negative, model might be untrained
            if max_logit < -5.0:
                print(f"\n  [WARN] All logits are very negative (< -5).")
                print(f"     This suggests the model weights might be from initialization (untrained).")
                print(f"     A trained model should have some positive logits.")
            elif max_logit > 5.0 and max_prob < 0.1:
                print(f"\n  [WARN] Logits are positive but probabilities are low.")
                print(f"     This is unusual. Checking sigmoid conversion...")
        
        # Create dictionary of all emotion probabilities
        all_probabilities = {}
        for i in range(self.num_labels):
            label = self.id_to_label[str(i)]
            all_probabilities[label] = float(probabilities[i])
        
        # Get all emotions with their scores
        all_emotion_scores = [
            (self.id_to_label[str(i)], float(probabilities[i]))
            for i in range(self.num_labels)
        ]
        
        # Sort by confidence (descending)
        all_emotion_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k emotions, but always include at least the top one
        # Use adaptive threshold: if max probability is very low, use relative threshold
        max_prob = all_emotion_scores[0][1] if all_emotion_scores else 0.0
        
        # If probabilities are very low (model might be untrained), use relative threshold
        if max_prob < 0.01:  # Less than 1%
            # Use relative threshold: top emotions that are at least 10% of max
            relative_threshold = max_prob * 0.1
            top_emotions = [
                (emotion, conf) for emotion, conf in all_emotion_scores
                if conf >= relative_threshold
            ][:top_k]
            # Always include at least the top emotion
            if not top_emotions and all_emotion_scores:
                top_emotions = [all_emotion_scores[0]]
        else:
            # Normal case: use absolute threshold
            top_emotions = [
                (emotion, conf) for emotion, conf in all_emotion_scores
                if conf >= threshold
            ][:top_k]
            # Always include at least the top emotion even if below threshold
            if not top_emotions and all_emotion_scores:
                top_emotions = [all_emotion_scores[0]]
        
        # Format results
        emotions = [
            {
                "emotion": emotion,
                "confidence": round(confidence, 4),
                "confidence_percent": round(confidence * 100, 2)
            }
            for emotion, confidence in top_emotions
        ]
        
        # Get top emotion (always return the highest, even if low)
        top_emotion = all_emotion_scores[0][0] if all_emotion_scores else None
        top_confidence = all_emotion_scores[0][1] if all_emotion_scores else 0.0
        
        # Check if model predictions are suspiciously low (might indicate untrained model)
        is_low_confidence = max_prob < 0.01  # Less than 1% max probability
        
        result = {
            "emotions": emotions,
            "top_emotion": top_emotion,
            "top_confidence": round(top_confidence, 4),
            "top_confidence_percent": round(top_confidence * 100, 2),
            "all_probabilities": {
                k: round(v, 4) for k, v in all_probabilities.items()
            },
            "warning": None
        }
        
        if is_low_confidence:
            result["warning"] = (
                "[WARN] All predictions are very low (<1%). "
                "The model may not be trained yet or needs retraining. "
                "Results shown are based on relative confidence."
            )
        
        return result
    
    def predict_batch(self, texts: List[str], top_k: int = 3, threshold: float = 0.05) -> List[Dict]:
        """
        Predict emotions for multiple texts.
        
        Args:
            texts: List of input texts
            top_k: Number of top emotions to return per text
            threshold: Minimum confidence threshold
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            result = self.predict(text, top_k=top_k, threshold=threshold)
            results.append(result)
        return results


def predict_emotion(text: str, model_dir: Path = MODEL_DIR, top_k: int = 3, threshold: float = 0.05) -> Dict:
    """
    Convenience function for quick predictions.
    
    Args:
        text: Input text to classify
        model_dir: Directory containing the trained model
        top_k: Number of top emotions to return
        threshold: Minimum confidence threshold
    
    Returns:
        Dictionary with prediction results
    """
    predictor = EmotionPredictor(model_dir)
    return predictor.predict(text, top_k=top_k, threshold=threshold)


def main():
    """
    Main function for interactive testing.
    """
    print("=" * 70)
    print("Emotion Classification - Prediction")
    print("=" * 70)
    
    try:
        # Initialize predictor
        print("\nInitializing emotion predictor...")
        predictor = EmotionPredictor()
        
        print("\n" + "=" * 70)
        print("Ready for predictions!")
        print("Enter text to classify emotions (type 'quit' to exit)")
        print("=" * 70)
        
        # Interactive loop
        while True:
            print("\n" + "-" * 70)
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            # Get prediction (using very low threshold to see all predictions)
            result = predictor.predict(text, top_k=7, threshold=0.001)
            
            # Display results
            print(f"\n[INPUT] Input: '{text}'")
            
            # Show warning if present
            if result.get('warning'):
                print(f"\n[WARN] {result['warning']}")
            
            print(f"\n[TOP] Top Emotion: {result['top_emotion']}")
            print(f"   Confidence: {result['top_confidence_percent']}%")
            
            if result['emotions']:
                print(f"\n[TOP] Top Emotions:")
                for i, emotion_data in enumerate(result['emotions'], 1):
                    print(f"   {i}. {emotion_data['emotion']}: {emotion_data['confidence_percent']}%")
            
            print(f"\n[PROBS] All Probabilities:")
            sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
            for emotion, prob in sorted_probs:
                bar_length = min(int(prob * 500), 50)  # Scale bar for visibility
                bar = "#" * bar_length
                print(f"   {emotion:12s}: {prob * 100:6.2f}% {bar}")
            
            # Diagnostic information if probabilities are suspiciously low
            max_prob = max(result['all_probabilities'].values())
            if max_prob < 0.1:  # Less than 10%
                print(f"\n[INFO] Diagnostic:")
                print(f"   Maximum probability: {max_prob * 100:.2f}%")
                print(f"   This is unusually low for a trained model.")
                print(f"   Your training showed 87% accuracy, so predictions should be much higher.")
                print(f"\n   Possible issues:")
                print(f"   1. Wrong checkpoint loaded - check if model_dir has checkpoint folders")
                print(f"   2. Model files incomplete - verify all files were downloaded from Colab")
                print(f"   3. Model path incorrect - current path: {MODEL_DIR}")
                print(f"\n   Verify your model directory structure:")
                print(f"   {MODEL_DIR}/")
                print(f"   ├── config.json (must exist)")
                print(f"   ├── pytorch_model.bin OR model.safetensors (must exist, ~260MB)")
                print(f"   ├── tokenizer_config.json")
                print(f"   └── vocab.txt")
                print(f"\n   If you see checkpoint-1/, checkpoint-2/, checkpoint-3/ folders:")
                print(f"   The script will automatically use the best checkpoint.")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nPlease ensure:")
        print("  1. The model is trained and saved in emotion_model/model/")
        print("  2. The model directory contains: config.json, pytorch_model.bin, tokenizer files")
        print("  3. label_map.json exists in emotion_model/model/ or emotion_model/")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

