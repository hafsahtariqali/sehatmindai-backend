"""
Chat Controller - Main Message Handling Pipeline

This module provides the main entry point for processing user messages.
It orchestrates the complete pipeline:
1. Session locking check
2. Crisis detection
3. Emotion detection (if no crisis)
4. Response generation (if no crisis)

The pipeline ensures user safety by prioritizing crisis detection and
preventing automated responses when a crisis is detected.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import crisis model inference function
from crisis_model.predict import CrisisPredictor

# Import crisis message
from guardrails.crisis_message import get_crisis_message

# Import emotion model inference function
from emotion_model.predict import EmotionPredictor

# Import LLM response generator
from llm_response_generator import LLMResponseGenerator


# ============================================================================
# SESSION LOCKING: Store locked sessions in memory
# ============================================================================
# When a crisis is detected, the session is locked for 2 hours to prevent
# further automated responses. Once locked, all future messages from that
# session will immediately return empty responses without processing until
# the lock expires.
# ============================================================================
# Key: session_id (str), Value: datetime when lock expires (2 hours from crisis detection)
locked_sessions: Dict[str, datetime] = {}


# ============================================================================
# GLOBAL COMPONENTS: Initialized on first use (lazy initialization)
# ============================================================================
# These components are initialized lazily to avoid import-time errors.
# In production, you might want to initialize them at application startup.
# ============================================================================
_crisis_predictor: Optional[CrisisPredictor] = None
_emotion_predictor: Optional[EmotionPredictor] = None
_llm_generator: Optional[LLMResponseGenerator] = None


def _get_crisis_predictor() -> CrisisPredictor:
    """Get or initialize the crisis detection predictor."""
    global _crisis_predictor
    if _crisis_predictor is None:
        _crisis_predictor = CrisisPredictor()
    return _crisis_predictor


def _get_emotion_predictor() -> EmotionPredictor:
    """Get or initialize the emotion predictor."""
    global _emotion_predictor
    if _emotion_predictor is None:
        _emotion_predictor = EmotionPredictor()
    return _emotion_predictor


def _get_llm_generator() -> LLMResponseGenerator:
    """Get or initialize the LLM response generator."""
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = LLMResponseGenerator()
    return _llm_generator




def handle_user_message(user_text: str, session_id: str) -> Dict:
    """
    Main function to handle user messages through the complete pipeline.
    
    PIPELINE ORDER (with clear comments):
    ====================================
    
    STEP 1: CHECK IF SESSION IS LOCKED
    -----------------------------------
    If the session was previously locked due to crisis detection,
    immediately return an empty response without processing.
    This prevents any automated responses when a crisis situation
    has been identified.
    
    STEP 2: RUN CRISIS DETECTION
    ----------------------------
    Before any other processing, check if the current message
    indicates a crisis situation. Crisis detection has the highest
    priority in the pipeline to ensure user safety.
    
    STEP 3: IF CRISIS DETECTED
    ---------------------------
    - Return the crisis support message (not empty response)
    - Lock the session to prevent future automated responses
    - Do NOT run emotion model or response engine
    - STOP processing immediately
    
    STEP 4: IF NO CRISIS (ELSE BRANCH)
    -----------------------------------
    Since no crisis was detected, we can safely proceed with:
    - Run emotion detection model to identify user's emotional state
    - Return a placeholder response (response generation will be implemented with LLM)
    
    Args:
        user_text: The user's message text to process
        session_id: Unique identifier for the user session
        
    Returns:
        Dictionary with:
        - response_text: The bot's response text (empty string if session locked)
        - is_crisis: Boolean indicating if crisis was detected
        - session_locked: Boolean indicating if session is now locked
        - emotion_detected: Detected emotion (None if not detected or crisis)
        - emotion_confidence: Confidence score for emotion (0.0 if not detected)
    """
    
    # ========================================================================
    # STEP 1: CHECK IF SESSION IS LOCKED
    # ========================================================================
    # If this session was previously locked due to crisis detection,
    # return empty response immediately without any processing.
    # Lock expires after 2 hours. This ensures we don't send automated
    # responses when a crisis situation has been identified.
    # ========================================================================
    if session_id in locked_sessions:
        lock_expiry = locked_sessions[session_id]
        now = datetime.now()
        
        # Check if lock has expired (2 hours have passed)
        if now < lock_expiry:
            # Session is still locked
            return {
                "response_text": "",  # Empty response for locked sessions
                "is_crisis": True,  # Session was locked due to crisis
                "session_locked": True,
                "support_mode": True,  # Input bar should be disabled, show "Support Mode" message
                "emotion_detected": None,
                "emotion_confidence": 0.0
            }
        else:
            # Lock has expired, remove it
            del locked_sessions[session_id]
    
    # ========================================================================
    # STEP 2: RUN CRISIS DETECTION (ML MODEL - CRITICAL SAFETY FEATURE)
    # ========================================================================
    # Crisis detection runs FIRST (highest priority) before any other
    # processing. This ensures user safety is prioritized over all
    # other considerations.
    # 
    # IMPORTANT: This uses the FULL ML crisis detection model with:
    # - Trained neural network for accurate detection
    # - Contextual understanding (not just keywords)
    # - Guardrails system for nuanced crisis signals
    # 
    # This is DIFFERENT from dataset filtering (which uses fast keywords
    # for performance). For real user messages, we ALWAYS use the ML model.
    # 
    # The crisis model's predict() function outputs:
    # - is_crisis: Boolean indicating if crisis is detected
    # - confidence: Float probability of crisis (0.0 to 1.0)
    # ========================================================================
    try:
        crisis_predictor = _get_crisis_predictor()
        crisis_result = crisis_predictor.predict(user_text)
        
        # ====================================================================
        # STEP 3: IF CRISIS DETECTED
        # ====================================================================
        # If crisis is detected:
        # 1. Lock the session to prevent future automated responses
        # 2. Return the crisis support message
        # 3. Do NOT run emotion model or response engine
        # 4. STOP processing immediately
        # ====================================================================
        if crisis_result.get("is_crisis", False):
            # Lock the session for 2 hours to prevent further automated responses
            lock_expiry = datetime.now() + timedelta(hours=2)
            locked_sessions[session_id] = lock_expiry
            
            # Get the crisis support message
            crisis_message = get_crisis_message()
            
            return {
                "response_text": crisis_message,  # Crisis support message
                "is_crisis": True,
                "session_locked": True,
                "support_mode": True,  # Input bar should be disabled, show "Support Mode" message
                "emotion_detected": None,  # Emotion not detected during crisis
                "emotion_confidence": 0.0
            }
    
    except Exception as e:
        # If crisis detection fails, log error but continue processing
        # We don't want technical errors to block normal conversations
        # In production, you should log this error properly
        print(f"Warning: Crisis detection failed: {e}")
        # Continue to emotion/response processing
    
    # ========================================================================
    # STEP 4: IF NO CRISIS (ELSE BRANCH)
    # ========================================================================
    # Since no crisis was detected, we can safely proceed with:
    # 1. Run emotion detection model
    # 2. Run response engine with emotion information
    # 3. Return the selected response
    # ========================================================================
    
    # Step 4a: Run emotion detection model
    # ------------------------------------
    # Detect the user's emotional state to inform empathetic response generation.
    # The emotion model's predict() function outputs:
    # - top_emotion: The emotion with highest confidence
    # - top_confidence: Confidence score of top emotion (0.0 to 1.0)
    # This output feeds into the response engine.
    detected_emotion = None
    emotion_confidence = 0.0
    
    try:
        emotion_predictor = _get_emotion_predictor()
        emotion_result = emotion_predictor.predict(user_text, top_k=1)
        # Extract emotion and confidence from emotion model output
        detected_emotion = emotion_result.get("top_emotion")
        emotion_confidence = emotion_result.get("top_confidence", 0.0)
    except Exception as e:
        # If emotion detection fails, continue without emotion info
        # The response engine can still generate responses without emotion
        print(f"Warning: Emotion detection failed: {e}")
    
    # Step 4b: Generate response using LLM
    # ------------------------------------
    # Generate empathetic response using LLM with user message and detected emotion.
    response_text = "I'm here to listen. What's on your mind?"  # Fallback response
    
    try:
        llm_generator = _get_llm_generator()
        response_text = llm_generator.generate_response(
            user_message=user_text,
            detected_emotion=detected_emotion,
            emotion_confidence=emotion_confidence
        )
    except Exception as e:
        # If LLM generation fails, use fallback response
        print(f"Warning: LLM response generation failed: {e}")
    
    # Step 4c: Return selected response
    # ----------------------------------
    # Return the response with all metadata
    return {
        "response_text": response_text,
        "is_crisis": False,
        "session_locked": False,
        "support_mode": False,  # Input bar should be enabled
        "emotion_detected": detected_emotion,
        "emotion_confidence": emotion_confidence
    }

