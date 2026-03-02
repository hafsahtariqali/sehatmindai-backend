"""
Test script with example statements for crisis detection.

Run this to see how the model responds to different types of messages.
"""

from predict import CrisisPredictor
import json


def test_crisis_detection():
    """Test crisis detection with various example statements."""
    
    print("=" * 70)
    print("Crisis Detection Model - Test Examples")
    print("=" * 70)
    
    # Initialize predictor
    print("\nInitializing crisis predictor...")
    try:
        predictor = CrisisPredictor()
        print("[OK] Model loaded successfully!\n")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Test cases
    test_cases = [
        # Clear crisis statements (should be detected)
        {
            "category": "CLEAR CRISIS - Should be detected",
            "examples": [
                "i want to kill myself",
                "i want to die",
                "i'm going to end my life",
                "i feel suicidal",
                "i want to hurt myself",
                "i'm thinking about suicide",
                "nothing matters anymore, i should just end it",
                "everyone would be better off if i was dead",
                "i'm going to overdose tonight",
                "i feel like cutting myself"
            ]
        },
        # Ambiguous crisis statements (might be detected)
        {
            "category": "AMBIGUOUS - May or may not be detected",
            "examples": [
                "i don't want to be here anymore",
                "i'm so hopeless",
                "everything feels pointless",
                "i can't take this anymore",
                "i just want the pain to stop",
                "i feel so alone and worthless",
                "what's the point of anything",
                "i give up"
            ]
        },
        # Non-crisis statements (should NOT be detected)
        {
            "category": "NOT CRISIS - Should NOT be detected",
            "examples": [
                "i feel sad today",
                "i'm having a bad day",
                "i feel stressed about work",
                "i'm worried about my exams",
                "i feel anxious about the presentation",
                "i'm feeling down",
                "i feel lonely sometimes",
                "i'm not feeling great today",
                "i had a rough day",
                "i'm feeling overwhelmed with work"
            ]
        },
        # Normal/positive statements (should NOT be detected)
        {
            "category": "NORMAL/POSITIVE - Should NOT be detected",
            "examples": [
                "i'm feeling good today",
                "i'm happy",
                "things are going well",
                "i'm okay",
                "just checking in",
                "how are you?",
                "i want to talk about my day",
                "i'm feeling better"
            ]
        }
    ]
    
    # Run tests
    results = {}
    
    for category_group in test_cases:
        category = category_group["category"]
        examples = category_group["examples"]
        
        print("\n" + "=" * 70)
        print(f"[{category}]")
        print("=" * 70)
        
        category_results = []
        
        for example in examples:
            result = predictor.predict(example)
            
            # Format result
            status = "[CRISIS]" if result['is_crisis'] else "[SAFE]"
            confidence_pct = result['confidence'] * 100
            
            print(f"\nText: \"{example}\"")
            print(f"  {status} Confidence: {result['confidence']:.4f} ({confidence_pct:.2f}%)")
            print(f"  Result: {json.dumps(result, indent=2)}")
            
            category_results.append({
                "text": example,
                "result": result
            })
        
        results[category] = category_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for category, examples in results.items():
        crisis_count = sum(1 for ex in examples if ex['result']['is_crisis'])
        avg_confidence = sum(ex['result']['confidence'] for ex in examples) / len(examples)
        
        print(f"\n{category}:")
        print(f"  Total examples: {len(examples)}")
        print(f"  Crisis detected: {crisis_count} ({crisis_count/len(examples)*100:.1f}%)")
        print(f"  Average confidence: {avg_confidence:.4f}")
    
    print("\n" + "=" * 70)
    print("Testing completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_crisis_detection()

