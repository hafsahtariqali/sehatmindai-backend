"""
Crisis Detection Module

This module detects potential mental health crises in user messages using:
1. Keyword-based detection (suicidal ideation, self-harm, etc.)
2. Emotion classifier predictions (extreme negative emotions)

IMPORTANT ETHICAL CONSIDERATIONS:
- This is a detection tool, not a diagnostic tool
- False positives are acceptable (better safe than sorry)
- Always err on the side of caution
- Crisis messages should direct users to professional help
- This module blocks further chatbot responses when crisis is detected
- Human intervention should be prioritized over automated responses

CRISIS RESOURCES (should be updated for your region):
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
"""

from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path
import json


class CrisisDetector:
    """
    Crisis detection system that combines keyword analysis and emotion detection.
    
    This class maintains state to block responses after a crisis is detected,
    ensuring the user receives appropriate resources and doesn't continue
    interacting with the chatbot during a crisis situation.
    """
    
    def __init__(self, emotion_classifier=None, blocked_sessions: Optional[set] = None):
        """
        Initialize the crisis detector.
        
        Args:
            emotion_classifier: Optional emotion classification model/pipeline.
                               If None, only keyword detection will be used.
            blocked_sessions: Set of session IDs that are currently blocked.
                             If None, creates a new set.
        """
        self.emotion_classifier = emotion_classifier
        self.blocked_sessions = blocked_sessions if blocked_sessions is not None else set()
        
        # Define crisis keywords - these indicate potential mental health emergencies
        # Organized by severity and category for clarity
        self.crisis_keywords = {
            # Direct suicidal ideation
            'suicide': [
                'suicide', 'kill myself', 'end my life', 'take my life',
                'end it all', 'not want to live', 'better off dead',
                'suicidal', 'commit suicide', 'end myself'
            ],
            # Self-harm indicators
            'self_harm': [
                'hurt myself', 'cut myself', 'self harm', 'self-harm',
                'cutting', 'burning myself', 'harm myself', 'self injury'
            ],
            # Immediate danger indicators
            'immediate_danger': [
                'going to kill myself', 'planning to end it', 'have a plan',
                'means to do it', 'ready to die', 'final goodbye',
                'this is the last', 'never see me again'
            ],
            # Hopelessness and despair (high risk indicators)
            'hopelessness': [
                'no point', 'nothing matters', 'no reason to live',
                'no way out', 'trapped', 'no hope', 'give up',
                'nothing left', 'no future'
            ]
        }
        
        # Compile regex patterns for efficient matching
        # Case-insensitive matching to catch variations
        self.crisis_patterns = {}
        for category, keywords in self.crisis_keywords.items():
            # Create pattern that matches whole words or phrases
            patterns = [r'\b' + re.escape(keyword) + r'\b' for keyword in keywords]
            self.crisis_patterns[category] = re.compile(
                '|'.join(patterns),
                re.IGNORECASE
            )
        
        # Define crisis message
        # This is a static message that will be shown when crisis is detected
        # It should direct users to professional help immediately
        self.crisis_message = (
            "I'm concerned about your safety. Please reach out for immediate help:\n\n"
            "🌐 National Suicide Prevention Lifeline: 988 (US)\n"
            "📱 Crisis Text Line: Text HOME to 741741 (US)\n"
            "🌍 International resources: https://www.iasp.info/resources/Crisis_Centres/\n\n"
            "You are not alone, and there are people who want to help you right now. "
            "Please contact a mental health professional or emergency services immediately.\n\n"
            "For your safety, I'm pausing our conversation. Please prioritize getting "
            "professional support."
        )
    
    def detect_keywords(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect crisis keywords in the text.
        
        This function searches for patterns that indicate potential mental health crises.
        We use multiple categories to catch different types of crisis indicators.
        
        Args:
            text: The user's message to analyze
        
        Returns:
            Tuple of (is_crisis_detected, list_of_matched_categories)
        """
        if not text or not isinstance(text, str):
            return False, []
        
        matched_categories = []
        
        # Check each category of crisis keywords
        for category, pattern in self.crisis_patterns.items():
            if pattern.search(text):
                matched_categories.append(category)
        
        # If any category matched, it's a potential crisis
        is_crisis = len(matched_categories) > 0
        
        return is_crisis, matched_categories
    
    def detect_emotion_crisis(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect crisis based on emotion classifier predictions.
        
        This function uses the emotion classifier to detect extreme negative emotions
        that might indicate a crisis situation. However, emotion alone is not sufficient
        - we combine it with keyword detection for more reliable results.
        
        Args:
            text: The user's message to analyze
        
        Returns:
            Tuple of (is_crisis_emotion, predicted_emotion)
        """
        if self.emotion_classifier is None:
            # If no emotion classifier provided, skip emotion-based detection
            return False, None
        
        try:
            # Get emotion prediction
            predictions = self.emotion_classifier(text)
            
            # Handle different return formats from emotion classifier
            if isinstance(predictions, list) and len(predictions) > 0:
                # If it's a list, get the first item (single prediction)
                if isinstance(predictions[0], list):
                    # Nested list format
                    emotion_scores = predictions[0]
                else:
                    # Direct list format
                    emotion_scores = predictions
            else:
                return False, None
            
            # Find the emotion with highest confidence
            top_emotion = max(emotion_scores, key=lambda x: x['score'])
            predicted_emotion = top_emotion['label']
            confidence = top_emotion['score']
            
            # Define emotions that might indicate crisis when combined with other factors
            # Note: These emotions alone don't indicate crisis, but can be warning signs
            crisis_emotions = ['sadness', 'anger', 'fear']
            
            # Only flag as crisis emotion if:
            # 1. The emotion is in our crisis emotions list
            # 2. The confidence is high (user is clearly expressing this emotion)
            is_crisis_emotion = (
                predicted_emotion.lower() in crisis_emotions and
                confidence > 0.7  # High confidence threshold
            )
            
            return is_crisis_emotion, predicted_emotion
            
        except Exception as e:
            # If emotion detection fails, don't block - just skip emotion check
            # We don't want technical errors to prevent crisis detection
            print(f"Warning: Emotion detection failed: {e}")
            return False, None
    
    def is_session_blocked(self, session_id: str) -> bool:
        """
        Check if a session is currently blocked due to crisis detection.
        
        Once a crisis is detected, the session is blocked to ensure the user
        receives appropriate resources and doesn't continue with automated responses.
        
        Args:
            session_id: Unique identifier for the user session
        
        Returns:
            True if session is blocked, False otherwise
        """
        return session_id in self.blocked_sessions
    
    def block_session(self, session_id: str) -> None:
        """
        Block a session from receiving further chatbot responses.
        
        This is called when a crisis is detected to ensure the user
        gets appropriate professional help rather than automated responses.
        
        Args:
            session_id: Unique identifier for the user session
        """
        self.blocked_sessions.add(session_id)
    
    def detect_crisis(
        self,
        text: str,
        session_id: str,
        require_both: bool = False
    ) -> Dict[str, any]:
        """
        Main crisis detection function.
        
        This function combines keyword detection and emotion detection to determine
        if a user message indicates a mental health crisis.
        
        Detection Logic:
        1. If session is already blocked, return crisis immediately
        2. Check for crisis keywords (primary indicator)
        3. Check for crisis emotions (secondary indicator)
        4. If require_both=False: Crisis if EITHER keyword OR emotion detected
        5. If require_both=True: Crisis only if BOTH keyword AND emotion detected
        
        We default to require_both=False because:
        - False positives are acceptable (better safe than sorry)
        - Keywords are more reliable indicators
        - We want to catch crises even if emotion detection fails
        
        Args:
            text: User's message to analyze
            session_id: Unique identifier for the user session
            require_both: If True, requires both keyword AND emotion detection.
                         If False (default), either one triggers crisis.
        
        Returns:
            Dictionary with:
            - is_crisis: Boolean indicating if crisis was detected
            - message: Crisis message if is_crisis=True, None otherwise
            - keyword_detected: Whether keywords were found
            - emotion_detected: Whether crisis emotions were detected
            - matched_categories: List of matched keyword categories
            - session_blocked: Whether this session is now blocked
        """
        # If session is already blocked, return crisis immediately
        if self.is_session_blocked(session_id):
            return {
                'is_crisis': True,
                'message': self.crisis_message,
                'keyword_detected': True,  # Already detected previously
                'emotion_detected': False,
                'matched_categories': ['previous_detection'],
                'session_blocked': True
            }
        
        # Step 1: Check for crisis keywords
        keyword_crisis, matched_categories = self.detect_keywords(text)
        
        # Step 2: Check for crisis emotions
        emotion_crisis, predicted_emotion = self.detect_emotion_crisis(text)
        
        # Step 3: Determine if crisis is detected based on detection mode
        if require_both:
            # Require BOTH keyword and emotion detection
            # This is more conservative and reduces false positives
            is_crisis = keyword_crisis and emotion_crisis
        else:
            # Require EITHER keyword OR emotion detection
            # This is more sensitive and catches more potential crises
            # We prioritize keyword detection as it's more reliable
            is_crisis = keyword_crisis or emotion_crisis
        
        # Step 4: If crisis detected, block the session
        if is_crisis:
            self.block_session(session_id)
            return {
                'is_crisis': True,
                'message': self.crisis_message,
                'keyword_detected': keyword_crisis,
                'emotion_detected': emotion_crisis,
                'predicted_emotion': predicted_emotion,
                'matched_categories': matched_categories,
                'session_blocked': True
            }
        else:
            # No crisis detected - normal processing can continue
            return {
                'is_crisis': False,
                'message': None,
                'keyword_detected': keyword_crisis,
                'emotion_detected': emotion_crisis,
                'predicted_emotion': predicted_emotion,
                'matched_categories': matched_categories,
                'session_blocked': False
            }
    
    def reset_session(self, session_id: str) -> None:
        """
        Reset a blocked session (unblock it).
        
        WARNING: This should only be used in exceptional circumstances,
        such as when a human moderator has reviewed the situation and
        determined it was a false positive.
        
        In production, this should require admin privileges.
        
        Args:
            session_id: Unique identifier for the user session
        """
        self.blocked_sessions.discard(session_id)
    
    def get_crisis_message(self) -> str:
        """
        Get the static crisis message.
        
        Returns:
            The crisis message that should be shown to users
        """
        return self.crisis_message


def create_crisis_detector(emotion_classifier=None) -> CrisisDetector:
    """
    Factory function to create a CrisisDetector instance.
    
    This is a convenience function that makes it easy to create
    a crisis detector with or without an emotion classifier.
    
    Args:
        emotion_classifier: Optional emotion classification pipeline/model
    
    Returns:
        Configured CrisisDetector instance
    """
    return CrisisDetector(emotion_classifier=emotion_classifier)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the crisis detector.
    
    This demonstrates how to use the crisis detector in your application.
    """
    
    print("=" * 70)
    print("Crisis Detector - Example Usage")
    print("=" * 70)
    
    # Create a crisis detector
    # In production, you would pass your emotion classifier here
    detector = create_crisis_detector(emotion_classifier=None)
    
    # Test cases
    test_cases = [
        ("I'm feeling a bit sad today", "session_1"),  # Normal - no crisis
        ("I want to kill myself", "session_2"),  # Crisis - keyword detected
        ("Everything is fine", "session_3"),  # Normal - no crisis
        ("I have no hope left", "session_4"),  # Crisis - keyword detected
        ("I'm planning to end it all tonight", "session_5"),  # Crisis - immediate danger
    ]
    
    print("\nTesting crisis detection:\n")
    
    for text, session_id in test_cases:
        print(f"Session: {session_id}")
        print(f"Message: '{text}'")
        
        result = detector.detect_crisis(text, session_id)
        
        print(f"  Crisis detected: {result['is_crisis']}")
        print(f"  Keyword detected: {result['keyword_detected']}")
        print(f"  Emotion detected: {result['emotion_detected']}")
        if result['matched_categories']:
            print(f"  Matched categories: {result['matched_categories']}")
        print(f"  Session blocked: {result['session_blocked']}")
        
        if result['is_crisis']:
            print(f"\n  Crisis message:\n  {result['message']}\n")
        
        print("-" * 70)
    
    # Test session blocking
    print("\nTesting session blocking:")
    print(f"Session 'session_2' blocked: {detector.is_session_blocked('session_2')}")
    print(f"Session 'session_3' blocked: {detector.is_session_blocked('session_3')}")

