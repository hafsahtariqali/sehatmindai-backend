"""
Prediction script for crisis detection model.

This script:
- Loads trained model and tokenizer from crisis_model/model/
- Accepts user text input
- Predicts probability of crisis
- Returns crisis detection result with confidence
"""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple
import numpy as np
# Import guardrails from the same directory (crisis_model)
# Use relative import to avoid conflicts with guardrails/ directory
from .guardrails import CrisisGuardrails, GuardrailResult


# Define paths
BASE_DIR = Path(__file__).parent  # crisis_model/
MODEL_DIR = BASE_DIR / "model"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
CRISIS_PHRASES_WHITELIST = BASE_DIR / "crisis_phrases_whitelist.json"
SAFE_PHRASES_WHITELIST = BASE_DIR / "safe_phrases_whitelist.json"
MAX_LENGTH = 128


class CrisisPredictor:
    """
    Crisis prediction class that loads and uses the trained model.
    """
    
    def __init__(self, model_dir: Path = MODEL_DIR, threshold: float = 0.75):
        """
        Initialize the crisis predictor.
        
        Args:
            model_dir: Directory containing the trained model
            threshold: Decision threshold for crisis classification (0.0 to 1.0)
        """
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.device = None
        self.crisis_phrases = []
        self.safe_phrases = []
        self.guardrails = None
        
        # Load components
        self._load_model()
        self._load_tokenizer()
        self._load_label_map()
        self._load_crisis_phrases_whitelist()
        self._load_safe_phrases_whitelist()
        self._load_guardrails()
    
    def _load_model(self):
        """Load the trained model."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                f"Please ensure the trained model is in this location."
            )
        
        # Find best checkpoint (same logic as evaluate.py)
        checkpoint_dirs = sorted(
            [d for d in self.model_dir.iterdir() 
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
                                    print(f"[OK] Found best checkpoint: {checkpoint_dir.name}")
                                    break
                    except:
                        pass
            
            if best_checkpoint is None:
                # Use latest checkpoint
                best_checkpoint = checkpoint_dirs[-1]
                print(f"[OK] Using latest checkpoint: {best_checkpoint.name}")
            
            model_path = best_checkpoint
        else:
            model_path = self.model_dir
            print(f"[OK] Using root model directory")
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.eval()
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        print(f"[OK] Model loaded on {self.device}")
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        print(f"[OK] Tokenizer loaded")
    
    def _load_label_map(self):
        """Load the label mapping."""
        if not LABEL_MAP_PATH.exists():
            # Fallback to BASE_DIR if not in MODEL_DIR
            label_map_path = BASE_DIR / "label_map.json"
        else:
            label_map_path = LABEL_MAP_PATH
        
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        print(f"[OK] Label map loaded: {self.label_map['num_labels']} labels")
        print(f"  Labels: {list(self.label_map['label_to_id'].keys())}")
    
    def _load_crisis_phrases_whitelist(self):
        """Load crisis phrases whitelist for safety override."""
        if not CRISIS_PHRASES_WHITELIST.exists():
            print(f"[WARN] Crisis phrases whitelist not found: {CRISIS_PHRASES_WHITELIST}")
            print(f"  Continuing without whitelist. Consider creating this file for safety.")
            self.crisis_phrases = []
            return
        
        try:
            with open(CRISIS_PHRASES_WHITELIST, 'r', encoding='utf-8') as f:
                whitelist_data = json.load(f)
            
            self.crisis_phrases = [phrase.lower().strip() for phrase in whitelist_data.get('phrases', [])]
            print(f"[OK] Loaded {len(self.crisis_phrases)} crisis phrases whitelist")
        except Exception as e:
            print(f"[WARN] Failed to load crisis phrases whitelist: {e}")
            self.crisis_phrases = []
    
    def _load_safe_phrases_whitelist(self):
        """Load safe phrases whitelist to prevent false positives."""
        if not SAFE_PHRASES_WHITELIST.exists():
            print(f"[INFO] Safe phrases whitelist not found: {SAFE_PHRASES_WHITELIST}")
            self.safe_phrases = []
            return
        
        try:
            with open(SAFE_PHRASES_WHITELIST, 'r', encoding='utf-8') as f:
                safe_data = json.load(f)
            
            self.safe_phrases = [phrase.lower().strip() for phrase in safe_data.get('phrases', [])]
            print(f"[OK] Loaded {len(self.safe_phrases)} safe phrases whitelist")
        except Exception as e:
            print(f"[WARN] Failed to load safe phrases whitelist: {e}")
            self.safe_phrases = []
    
    def _load_guardrails(self):
        """Load intelligent guardrails system."""
        try:
            guardrails_config_path = BASE_DIR / "guardrails_config.json"
            self.guardrails = CrisisGuardrails(guardrails_config_path)
            print(f"[OK] Guardrails system loaded")
        except Exception as e:
            print(f"[WARN] Failed to load guardrails: {e}")
            print(f"  Continuing without guardrails (safety features limited)")
            self.guardrails = None
    
    def _check_crisis_whitelist(self, text: str) -> Tuple[bool, str]:
        """
        Check if text contains any crisis phrases from whitelist.
        
        Args:
            text: Input text to check
        
        Returns:
            Tuple of (is_crisis, matched_phrase)
            - is_crisis: True if any phrase found, False otherwise
            - matched_phrase: The phrase that was matched (empty string if none)
        """
        if not self.crisis_phrases or not text:
            return False, ""
        
        text_lower = text.lower()
        
        # Check if any phrase is in the text
        for phrase in self.crisis_phrases:
            if phrase in text_lower:
                return True, phrase
        
        return False, ""
    
    def predict(self, text: str) -> Dict:
        """
        Predict crisis probability for given text.
        
        Args:
            text: Input text to classify
        
        Returns:
            Dictionary containing:
            - is_crisis: Boolean indicating if crisis is detected
            - confidence: Float probability of crisis (0.0 to 1.0)
        """
        if not text or not text.strip():
            return {
                "is_crisis": False,
                "confidence": 0.0,
                "overridden_by_whitelist": False,
                "overridden_by_guardrails": False
            }
        
        # SAFETY LAYER 0: Check safe phrases whitelist first (prevent false positives)
        # This overrides everything for common non-crisis phrases (e.g., academic stress)
        text_lower = text.lower()
        for safe_phrase in self.safe_phrases:
            if safe_phrase in text_lower:
                print(f"\n[SAFE] Safe phrase detected: '{safe_phrase}'")
                print(f"[SAFE] Marking as NOT crisis (false positive prevention)")
                return {
                    "is_crisis": False,
                    "confidence": 0.0,
                    "overridden_by_whitelist": False,
                    "overridden_by_guardrails": False,
                    "overridden_by_safe_whitelist": True,
                    "matched_safe_phrase": safe_phrase
                }
        
        # SAFETY LAYER 1: Check crisis phrases whitelist
        # This overrides model prediction for safety-critical phrases
        is_in_whitelist, matched_phrase = self._check_crisis_whitelist(text)
        
        if is_in_whitelist:
            print(f"\n[SAFETY] Crisis phrase detected in whitelist: '{matched_phrase}'")
            print(f"[SAFETY] Overriding model prediction - ALWAYS marking as CRISIS")
            return {
                "is_crisis": True,
                "confidence": 0.99,  # High confidence for whitelist matches
                "overridden_by_whitelist": True,
                "overridden_by_guardrails": False,
                "matched_phrase": matched_phrase
            }
        
        # SAFETY LAYER 2: Intelligent guardrails (contextual analysis)
        # This detects patterns and combinations that indicate crisis
        guardrail_result = None
        if self.guardrails:
            guardrail_result = self.guardrails.analyze(text)
            
            if guardrail_result.is_crisis and guardrail_result.severity in ["critical", "high"]:
                print(f"\n[GUARDRAILS] Crisis detected by intelligent guardrails")
                print(f"[GUARDRAILS] Severity: {guardrail_result.severity.upper()}")
                print(f"[GUARDRAILS] Risk Score: {guardrail_result.risk_score:.2%}")
                if guardrail_result.reasons:
                    print(f"[GUARDRAILS] Reason: {guardrail_result.reasons[0]}")
                if guardrail_result.triggered_rules:
                    print(f"[GUARDRAILS] Triggered rules: {', '.join(guardrail_result.triggered_rules)}")
                
                # Override model prediction for critical/high severity
                return {
                    "is_crisis": True,
                    "confidence": guardrail_result.confidence,
                    "overridden_by_whitelist": False,
                    "overridden_by_guardrails": True,
                    "guardrail_severity": guardrail_result.severity,
                    "guardrail_reasons": guardrail_result.reasons,
                    "risk_score": guardrail_result.risk_score
                }
        
        # Tokenize input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply softmax to get probabilities (binary classification with 2 classes)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Extract probabilities for both classes
        not_crisis_prob = float(probabilities[0])  # Probability of not_crisis (label 0)
        crisis_probability = float(probabilities[1])  # Probability of crisis (label 1)
        
        # Get predicted class (argmax)
        predicted_class_id = int(np.argmax(probabilities))
        predicted_label_name = self.label_map['id_to_label'][str(predicted_class_id)]
        
        # Debug: Print raw logits and probabilities
        #print(f"\n[DEBUG] Raw logits: {logits.cpu().numpy()[0]}")
        #print(f"[DEBUG] Probabilities: not_crisis={not_crisis_prob:.4f} (class 0), crisis={crisis_probability:.4f} (class 1)")
        #print(f"[DEBUG] Predicted class: {predicted_class_id} ({predicted_label_name})")
        #print(f"[DEBUG] Label map: {self.label_map['id_to_label']}")
        
        # Standard interpretation: class 0 = not_crisis, class 1 = crisis
        # Determine if crisis based on threshold
        is_crisis = crisis_probability >= self.threshold
        
        # Combine model prediction with guardrail analysis
        final_is_crisis = bool(is_crisis)
        final_confidence = round(crisis_probability, 4)
        
        # If guardrails detected medium risk, only boost model confidence if it's already positive
        # We DON'T activate crisis mode for medium risk - only critical/high severity trigger crisis mode
        if guardrail_result and guardrail_result.severity == "medium":
            if is_crisis:
                # Boost confidence slightly if model already detected crisis
                final_confidence = min(1.0, crisis_probability + 0.1)
            # Removed: No longer forcing crisis mode for medium severity guardrails
            # Medium severity is informational only - doesn't override model decision
        
        return {
            "is_crisis": final_is_crisis,
            "confidence": final_confidence,
            "overridden_by_whitelist": False,
            "overridden_by_guardrails": False,  # Only critical/high severity override, not medium
            "risk_score": guardrail_result.risk_score if guardrail_result else None
        }


def main():
    """Interactive prediction function."""
    print("=" * 70)
    print("Crisis Detection - Prediction")
    print("=" * 70)
    
    # Initialize predictor
    print("\nInitializing crisis predictor...")
    try:
        predictor = CrisisPredictor()
        print("\n[OK] Ready for predictions!")
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize predictor: {e}")
        return
    
    print("\n" + "=" * 70)
    print("Enter text to check for crisis (type 'quit' to exit)")
    print("=" * 70)
    
    while True:
        try:
            print("\n" + "-" * 70)
            text = input("Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not text:
                print("[WARN] Empty input. Please enter some text.")
                continue
            
            # Predict
            result = predictor.predict(text)
            
            # Display result
            print(f"\n[INPUT] Text: '{text}'")
            print(f"\n[RESULT]")
            print(f"  Crisis Detected: {result['is_crisis']}")
            print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            
            if result.get('overridden_by_whitelist', False):
                print(f"  [SAFETY] Overridden by crisis phrases whitelist")
                if 'matched_phrase' in result:
                    print(f"  Matched phrase: '{result['matched_phrase']}'")
            
            if result.get('overridden_by_guardrails', False):
                print(f"  [GUARDRAILS] Enhanced by intelligent guardrails")
                if 'guardrail_severity' in result:
                    print(f"  Severity: {result['guardrail_severity'].upper()}")
                if 'guardrail_reasons' in result and result['guardrail_reasons']:
                    print(f"  Reason: {result['guardrail_reasons'][0]}")
                if 'risk_score' in result and result['risk_score']:
                    print(f"  Risk Score: {result['risk_score']:.2%}")
            
            if result['is_crisis']:
                print(f"\n  [WARN] CRISIS DETECTED!")
                print(f"  This message indicates potential crisis situation.")
                print(f"  Immediate intervention recommended.")
            else:
                print(f"\n  [OK] No crisis detected.")
                print(f"  Message appears to be safe.")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] Prediction error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
