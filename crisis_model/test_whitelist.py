"""
Quick test script to verify crisis phrases whitelist is working.
Tests the phrases the user mentioned that should trigger crisis detection.
"""

import json
from pathlib import Path
from predict import CrisisPredictor

# Test phrases from user
test_phrases = [
    "ending myself",
    "im thinking about suicide",
    "i'm thinking about suicide",
    "thinking about suicide",
    "i want to kill myself",
    "i am so happy",  # Should NOT trigger whitelist (negative test)
    "my day is going well"  # Should NOT trigger whitelist (negative test)
]

def test_whitelist():
    """Test the crisis phrases whitelist."""
    print("=" * 70)
    print("Testing Crisis Phrases Whitelist")
    print("=" * 70)
    
    # Load whitelist directly to show what phrases are loaded
    whitelist_path = Path(__file__).parent / "crisis_phrases_whitelist.json"
    if whitelist_path.exists():
        with open(whitelist_path, 'r', encoding='utf-8') as f:
            whitelist_data = json.load(f)
        print(f"\nLoaded {len(whitelist_data['phrases'])} phrases in whitelist")
        print(f"First 5 phrases: {whitelist_data['phrases'][:5]}")
    
    # Initialize predictor (this will load the whitelist)
    print("\n" + "-" * 70)
    print("Initializing predictor...")
    try:
        predictor = CrisisPredictor()
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return
    
    print(f"\nWhitelist loaded: {len(predictor.crisis_phrases)} phrases")
    
    # Test each phrase
    print("\n" + "=" * 70)
    print("Testing Phrases")
    print("=" * 70)
    
    for phrase in test_phrases:
        print(f"\n{'-' * 70}")
        print(f"Testing: '{phrase}'")
        
        # Check whitelist directly
        is_whitelisted, matched = predictor._check_crisis_whitelist(phrase)
        
        print(f"  Whitelist check: {is_whitelisted}")
        if matched:
            print(f"  Matched phrase: '{matched}'")
        
        # Get full prediction (includes whitelist override)
        result = predictor.predict(phrase)
        
        print(f"  Prediction result:")
        print(f"    Crisis: {result['is_crisis']}")
        print(f"    Confidence: {result['confidence']:.2%}")
        if result.get('overridden_by_whitelist', False):
            print(f"    [SAFETY OVERRIDE] Overridden by whitelist!")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_whitelist()

