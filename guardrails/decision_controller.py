"""
Decision Controller for Crisis Detection

This module controls the flow of user messages by checking for crisis situations.
It uses the crisis detection model to determine if a response should be allowed.

This file only controls flow - it does not generate responses.
"""

import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path to import from crisis_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from crisis_model.predict import CrisisPredictor


class DecisionController:
    """
    Controller that determines whether to allow responses based on crisis detection.
    """
    
    def __init__(self):
        """Initialize the decision controller with the crisis detection model."""
        self.predictor = CrisisPredictor()
    
    def check(self, user_text: str) -> Dict:
        """
        Check if response should be allowed based on crisis detection.
        
        Args:
            user_text: The user's input text to check
            
        Returns:
            Dictionary with:
            - allow_response: True if response should be allowed, False if crisis detected
            - crisis: True if crisis is detected, False otherwise
            - detection_result: Full detection result from the model (for logging)
        """
        if not user_text or not user_text.strip():
            return {
                "allow_response": True,
                "crisis": False,
                "detection_result": None
            }
        
        # Call the crisis detection model
        result = self.predictor.predict(user_text)
        
        # Extract crisis status from the prediction result
        is_crisis = result.get("is_crisis", False)
        
        # Return decision format with full detection result for logging
        return {
            "allow_response": not is_crisis,
            "crisis": is_crisis,
            "detection_result": result  # Include full result for logging
        }


def check_crisis(user_text: str) -> Dict[str, bool]:
    """
    Convenience function to check crisis status.
    
    Creates a DecisionController instance and checks the user text.
    
    Args:
        user_text: The user's input text to check
        
    Returns:
        Dictionary with:
        - allow_response: True if response should be allowed, False if crisis detected
        - crisis: True if crisis is detected, False otherwise
    """
    controller = DecisionController()
    return controller.check(user_text)

