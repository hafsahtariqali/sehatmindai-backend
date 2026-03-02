"""
Developer Test Script for Chat Controller

This script tests the handle_user_message function with various test cases.
It prints the response, detected emotion, and confidence scores.

This is for developer testing only - not for production use.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from chat_controller import handle_user_message


def test_message(user_text: str, session_id: str = "test_session"):
    """
    Test a single message and print the results.
    
    Args:
        user_text: The message to test
        session_id: Session ID for the test (default: "test_session")
    """
    print("=" * 80)
    print(f"TEST MESSAGE")
    print("=" * 80)
    print(f"User Text: '{user_text}'")
    print(f"Session ID: {session_id}")
    print("-" * 80)
    
    # Call handle_user_message
    result = handle_user_message(user_text, session_id)
    
    # Print results
    print("\nRESULT:")
    print(f"  Response Text: {result.get('response_text', '')[:200]}...")
    if len(result.get('response_text', '')) > 200:
        print(f"  (Response truncated - full length: {len(result.get('response_text', ''))} chars)")
    print(f"  Is Crisis: {result.get('is_crisis', False)}")
    print(f"  Session Locked: {result.get('session_locked', False)}")
    print(f"  Detected Emotion: {result.get('emotion_detected', None)}")
    print(f"  Emotion Confidence: {result.get('emotion_confidence', 0.0):.4f}")
    print("=" * 80)
    print()


def main():
    """Run test cases."""
    print("\n" + "=" * 80)
    print("CHAT CONTROLLER - DEVELOPER TEST")
    print("=" * 80)
    print("\nThis script tests the handle_user_message function.")
    print("Initializing models may take a moment on first run...\n")
    
    # Test case 1: Normal message
    print("\n[TEST 1] Normal message - should detect emotion and generate response")
    test_message("I feel really overwhelmed", session_id="test_session_1")
    
    # Test case 2: Potentially concerning message
    print("\n[TEST 2] Concerning message - should detect emotion and generate empathetic response")
    test_message("Nothing feels worth it anymore", session_id="test_session_2")
    
    # Test case 3: Another normal message
    print("\n[TEST 3] Another normal message")
    test_message("I'm feeling anxious about my upcoming exam", session_id="test_session_3")
    
    # Test case 4: Sad message
    print("\n[TEST 4] Sad message")
    test_message("I've been feeling really down lately", session_id="test_session_4")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("\nNote: If a crisis is detected, the session will be locked.")
    print("Subsequent messages from the same session will return empty responses.")
    print()


if __name__ == "__main__":
    main()

