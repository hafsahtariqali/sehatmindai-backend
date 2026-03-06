"""
FastAPI Server for Mental Health Chatbot

This server provides a REST API that:
- Accepts user messages via POST requests
- Processes messages through crisis detection and emotion detection
- Generates responses using LLM (to be implemented)
- Returns chatbot responses as JSON

This API is designed to be consumed by a Flutter mobile application.
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
import httpx
import asyncio
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import crisis detection decision controller
from guardrails.decision_controller import DecisionController

# Import crisis logging service for anonymized research logging
from guardrails.crisis_logging import log_crisis_detection

# Import crisis message
from guardrails.crisis_message import get_crisis_message, get_crisis_message_urdu

# REMOVED: Intro message imports - chatbot no longer sends greetings

# Import user data service for fetching full names from Firestore
from services.user_data_service import get_user_full_name, get_user_preferred_language

# Import email and OTP services
from services.email_service import EmailService
from services.otp_service import OTPService

# Import rate limiter
from guardrails.rate_limiter import check_rate_limit

# Import signup guard for spam account protection
from guardrails.signup_guard import check_signup_guard

# Import performance optimizer
from guardrails.performance_optimizer import get_cached_user_data, cache_user_data, cleanup_expired_cache

# Import emotion model
from emotion_model.predict import EmotionPredictor

# Import LLM response generator
from llm_response_generator import LLMResponseGenerator

# Import translation for Urdu/Roman Urdu support (async for LLM-based translation)
from translation_service import (
    process_user_message_for_pipeline_async,
    process_bot_response_for_user_async,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SehatMind Chatbot API",
    description="API for mental health chatbot",
    version="1.0.0"
)

# Configure CORS for Flutter app
# Flutter apps need CORS enabled to make requests from mobile devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests"""
    logger.info(f"→ Incoming request: {request.method} {request.url.path}")
    logger.info(f"  Query params: {dict(request.query_params)}")
    response = await call_next(request)
    logger.info(f"← Response: {response.status_code} for {request.method} {request.url.path}")
    return response


# Configuration
# Crisis logging configuration
# Set to True to use Firebase for logging (requires Firebase Admin SDK)
# Set to False to use file-based logging
USE_FIREBASE_LOGGING = os.getenv("USE_FIREBASE_LOGGING", "false").lower() == "true"

# ============================================================================
# SAFETY GUARDRAIL: Conversation Locking for Crisis Detection
# ============================================================================
# When a crisis is detected, the conversation is locked (conversation_locked = True).
# Once locked, all future messages from that session will immediately return
# empty responses without calling emotion or response models.
# This ensures user safety by preventing further automated interactions
# when a crisis situation has been identified.
# ============================================================================

# Store locked sessions in memory
# Key: session_id, Value: datetime when lock expires (2 hours from crisis detection)
locked_sessions: Dict[str, datetime] = {}


def cleanup_expired_locks():
    """
    Remove all expired session locks from memory.
    This helps keep the locked_sessions dictionary clean.
    """
    now = datetime.now()
    expired_sessions = [session_id for session_id, expiry in locked_sessions.items() if now >= expiry]
    for session_id in expired_sessions:
        del locked_sessions[session_id]
    if expired_sessions:
        logger.info(f"Cleaned up {len(expired_sessions)} expired session lock(s)")

# Store conversation history in memory
# Key: session_id, Value: List of message dicts with 'role' and 'content'
# Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
conversation_history: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_LENGTH = 10  # Keep last 10 messages (5 user + 5 assistant pairs)

# REMOVED: sessions_with_intro tracking - chatbot no longer sends greetings

# Global crisis detection controller (initialized at startup)
crisis_controller: Optional[DecisionController] = None

# Global emotion predictor (initialized at startup)
emotion_predictor: Optional[EmotionPredictor] = None

# Global LLM response generator (initialized at startup)
llm_generator: Optional[LLMResponseGenerator] = None


# Request/Response Models
# These define the structure of data sent to and received from the API

class ChatMessage(BaseModel):
    """
    Model for incoming chat messages from Flutter app.
    
    Fields:
        message: The user's message text
        sender_id: Unique identifier for the user/session
        user_name: Optional user's first name for personalization
        preferred_language: Optional preferred language ("English" or "Urdu")
    """
    message: str = Field(..., description="The user's message", min_length=1, max_length=1000)
    sender_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the user session. If not provided, a new session is created."
    )
    user_name: Optional[str] = Field(
        default=None,
        description="User's first name for personalization (optional)"
    )
    preferred_language: Optional[str] = Field(
        default=None,
        description="User's preferred language: 'English' or 'Urdu' (optional)"
    )


class BotResponse(BaseModel):
    """
    Model for bot responses sent back to Flutter app.
    
    Fields:
        text: The bot's response message
        sender: Always "bot" to identify the sender
    """
    text: str
    sender: str = "bot"


class ChatResponse(BaseModel):
    """
    Complete response model for chat API endpoint.
    
    Fields:
        responses: List of bot responses (usually one, but can be multiple)
        session_id: The session identifier used for this conversation
        timestamp: When the response was generated
        session_locked: Boolean indicating if session is locked (input should be disabled)
        support_mode: Boolean indicating if in support mode (same as session_locked, for clarity)
        response: Singular response text for Flutter app compatibility
        crisis: Boolean indicating if crisis was detected
        emotion: Detected emotion string
    """
    responses: List[BotResponse]
    session_id: str
    timestamp: str
    session_locked: bool = False
    support_mode: bool = False  # Alias for session_locked - when True, input bar should be disabled
    response: Optional[str] = None  # Singular response for Flutter compatibility
    crisis: bool = False  # Crisis detection flag for Flutter
    emotion: Optional[str] = None  # Detected emotion for Flutter


class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Fields:
        status: API server status
        timestamp: Current server time
    """
    status: str
    timestamp: str


class PersonalizeContentRequest(BaseModel):
    """
    Request model for content personalization endpoint.
    
    Fields:
        mood: Current mood (0-4, where 0 is very sad, 4 is very happy)
        focus_areas: List of focus areas/modules the user has selected
        recently_shown_lessons: List of lesson IDs shown recently (to avoid repetition)
        recently_shown_activities: List of activity IDs shown recently (to avoid repetition)
        available_content: List of available content items with metadata
    """
    mood: int = Field(..., ge=0, le=4, description="Current mood (0-4)")
    focus_areas: List[str] = Field(..., description="User's selected focus areas")
    recently_shown_lessons: List[str] = Field(default_factory=list, description="Recently shown lesson IDs")
    recently_shown_activities: List[str] = Field(default_factory=list, description="Recently shown activity IDs")
    available_content: List[Dict[str, Any]] = Field(..., description="Available content metadata")


class PersonalizeContentResponse(BaseModel):
    """
    Response model for content personalization endpoint.
    
    Fields:
        selected_lessons: List of 3 selected lesson IDs
        selected_activity: 1 selected activity ID
    """
    selected_lessons: List[str] = Field(..., description="3 selected lesson IDs")
    selected_activity: Optional[str] = Field(None, description="1 selected activity ID")


class SendOTPRequest(BaseModel):
    """Request model for sending OTP"""
    email: str = Field(..., description="User's email address")


class SendOTPResponse(BaseModel):
    """Response model for sending OTP"""
    success: bool
    message: str


class VerifyOTPRequest(BaseModel):
    """Request model for verifying OTP"""
    email: str = Field(..., description="User's email address")
    otp: str = Field(..., description="OTP code to verify", min_length=6, max_length=6)


class VerifyOTPResponse(BaseModel):
    """Response model for verifying OTP"""
    success: bool
    message: str


class ResetPasswordRequest(BaseModel):
    """Request model for resetting password"""
    email: str = Field(..., description="User's email address")
    new_password: str = Field(..., description="New password", min_length=6)


class ResetPasswordResponse(BaseModel):
    """Response model for resetting password"""
    success: bool
    message: str


class SignupGuardRequest(BaseModel):
    """Request model for signup spam protection"""
    email: str = Field(..., description="User's email address")
    device_id: Optional[str] = Field(
        default=None,
        description="Optional device identifier from client (for per-device limits)"
    )


class SignupGuardResponse(BaseModel):
    """Response model for signup spam protection"""
    allowed: bool
    reason: Optional[str] = Field(
        default=None,
        description="Reason if signup is blocked or additional info if allowed"
    )


# Helper Functions

def generate_session_id() -> str:
    """
    Generate a unique session ID for new users.
    
    In production, you might want to use a more sophisticated
    session management system (e.g., JWT tokens, database sessions).
    
    Returns:
        A unique session identifier string
    """
    import uuid
    return f"user_{uuid.uuid4().hex[:12]}"


# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """
    Root endpoint - health check.
    
    This endpoint can be used to verify the API is running.
    
    Returns:
        Health status information
    """
    return HealthResponse(
        status="running",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Same as root endpoint, but more explicit for health monitoring.
    
    Returns:
        Health status information
    """
    return await root()


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Main chat endpoint - receives user messages and returns bot responses.
    
    This is the primary endpoint that the Flutter app will use.
    
    SAFETY HIERARCHY - Decision Flow:
    1. Receive user message and optional session ID
    2. Generate session ID if not provided
    3. SAFETY GUARDRAIL: Check if conversation is locked (crisis detected previously)
    4. SAFETY GUARDRAIL: Crisis model runs FIRST (highest priority)
       - If crisis detected: Send crisis message, lock conversation, STOP
    5. If no crisis: Emotion model runs
    6. If no crisis: LLM response generation (to be implemented)
    7. Format and return response
    
    This safety hierarchy ensures:
    - Crisis detection always has priority (user safety first)
    - Response generation only runs when safe (no crisis)
    - Emotion detection informs empathetic responses
    - Crisis messages are sent immediately, preventing inappropriate responses
    
    Args:
        message: ChatMessage object containing user message and optional sender_id
    
    Returns:
        ChatResponse with bot responses, session ID, and timestamp
        Returns crisis message if crisis detected
        Returns empty response if conversation is locked
    
    Raises:
        HTTPException: If message is invalid or models are unavailable
    """
    try:
        return await _process_chat_message(message)
    except Exception as e:
        logger.error(f"Unhandled error in chat endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Generate a session ID for error response
        session_id = message.sender_id or generate_session_id()
        # Return error response to prevent Railway timeout
        fallback_text = "I'm here to listen. What's on your mind?"
        return ChatResponse(
            responses=[BotResponse(text=fallback_text, sender="bot")],
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            session_locked=False,
            support_mode=False,
            response=fallback_text,
            crisis=False,
            emotion=None
        )


async def _process_chat_message(message: ChatMessage) -> ChatResponse:
    """
    Internal function to process chat messages.
    """
    # Reduced logging for performance (only log key events)
    logger.info(f"Chat request: sender_id={message.sender_id}, msg_len={len(message.message)}, message_preview={message.message[:50]}")
    
    # Validate message
    if not message.message or not message.message.strip():
        logger.error("ERROR: Empty message received")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    
    # Get or generate session ID
    session_id = message.sender_id or generate_session_id()
    logger.info(f"Using session_id: {session_id} (was sender_id provided: {message.sender_id is not None})")
    
    # Clean up expired locks and cache periodically (runs on every request)
    cleanup_expired_locks()
    # Clean cache every 100 requests to avoid overhead
    if len(conversation_history) % 100 == 0:
        cleanup_expired_cache()
    
    # ========================================================================
    # URDU/ROMAN URDU: Detect language first (needed for rate limit messages)
    # ========================================================================
    # We need to detect language early for localized rate limit messages
    # ========================================================================
    # Quick language detection for rate limit messages (before full translation)
    import re
    _URDU_SCRIPT_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    user_wrote_urdu_quick = bool(_URDU_SCRIPT_PATTERN.search(message.message)) or any(
        word in message.message.lower() for word in ['mujhe', 'main', 'hai', 'hoon', 'ka', 'ki', 'se', 'ko']
    )
    
    # ========================================================================
    # RATE LIMITING: Check if user has exceeded message limits
    # ========================================================================
    # This prevents exceeding Groq API free tier limits and ensures fair usage.
    # Limits are configurable via environment variables.
    # Messages are localized based on user's language.
    # ========================================================================
    is_allowed, rate_limit_error = check_rate_limit(session_id, user_wrote_urdu_quick)
    if not is_allowed:
        logger.warning(f"Rate limit exceeded for session {session_id}")
        # Return rate limit error message (already localized)
        error_response_text = rate_limit_error or ("معاف کیجیے، براہ کرم انتظار کریں۔" if user_wrote_urdu else "Please wait before sending another message.")
        return ChatResponse(
            responses=[BotResponse(text=error_response_text, sender="bot")],
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            session_locked=False,
            support_mode=False,
            response=error_response_text,
            crisis=False,
            emotion=None
        )
    
    # ========================================================================
    # URDU/ROMAN URDU: Translate to English for pipeline if needed (LLM-based)
    # ========================================================================
    # If user writes in Urdu or Roman Urdu, translate to English first.
    # Pipeline (crisis, emotion, LLM) works in English for best accuracy.
    # We'll translate the bot's response back to Urdu before sending.
    # ========================================================================
    text_for_pipeline, user_wrote_urdu = await process_user_message_for_pipeline_async(message.message)
    original_urdu_message = message.message if user_wrote_urdu else None  # Store original for Urdu responses
    logger.debug(f"Language detected: {'Urdu' if user_wrote_urdu else 'English'}")
    
    # ========================================================================
    # REMOVED: Intro message system - frontend handles greetings
    # ========================================================================
    # The frontend shows the automated greeting "Hi [name], how is your mood today?"
    # when a new session starts. The chatbot should NEVER send greetings.
    # Chatbot only responds to user messages - NO GREETINGS.
    # ========================================================================
    
    # Initialize conversation history if it doesn't exist
    # PRE-SEED with the greeting that the frontend already shows.
    # This tells the LLM "the greeting already happened" so it won't generate one.
    if session_id not in conversation_history:
        # Get user's name for the greeting (matches what frontend shows)
        user_name = message.user_name or "there"
        conversation_history[session_id] = [
            {"role": "assistant", "content": f"Hi {user_name}, how is your mood today?"}
        ]
        logger.info(f"New session {session_id} - pre-seeded history with frontend greeting")
    
    # ========================================================================
    # SAFETY GUARDRAIL: Check if conversation is already locked
    # ========================================================================
    # If a crisis was detected in a previous message, the conversation is locked
    # for 2 hours. Once locked, we immediately return an empty response without
    # calling any emotion or response models. This prevents further automated
    # interactions when a crisis situation has been identified.
    # ========================================================================
    if session_id in locked_sessions:
        lock_expiry = locked_sessions[session_id]
        now = datetime.now()
        
        # Check if lock has expired (2 hours have passed)
        if now < lock_expiry:
            # Session is still locked
            logger.warning(f"Session {session_id} is locked due to crisis detection (expires at {lock_expiry.isoformat()}) - returning empty response")
            # Clear conversation history for locked sessions
            if session_id in conversation_history:
                conversation_history[session_id] = []
            # Return empty response immediately - do not process message or call any models
            # Set session_locked and support_mode to True so Flutter app can disable input bar
            return ChatResponse(
                responses=[],  # Empty response list
                session_id=session_id,
                timestamp=now.isoformat(),
                session_locked=True,
                support_mode=True,  # Input bar should be disabled, show "Support Mode" message
                response="",  # Empty response for Flutter
                crisis=True,  # Session locked due to crisis
                emotion=None
            )
        else:
            # Lock has expired, remove it
            logger.info(f"Session {session_id} lock has expired - removing lock")
            del locked_sessions[session_id]
    
    # ========================================================================
    # SAFETY GUARDRAIL: Check for crisis in current message
    # ========================================================================
    # Before calling any emotion/response models,
    # we check if the current message indicates a crisis situation.
    # If crisis is detected, we lock the conversation and return crisis message.
    # ========================================================================
    crisis_check_result = None
    crisis_level = "none"
    if crisis_controller is not None:
        try:
            crisis_check_result = crisis_controller.check(text_for_pipeline)
            
            if crisis_check_result.get("crisis", False):
                # Crisis detected - lock the conversation for 2 hours
                lock_expiry = datetime.now() + timedelta(hours=2)
                locked_sessions[session_id] = lock_expiry
                logger.warning(f"Crisis detected for session {session_id} - locking conversation until {lock_expiry.isoformat()}")
                
                # ========================================================================
                # ANONYMIZED RESEARCH LOGGING: Log crisis detection event
                # ========================================================================
                # This logs crisis detection events for research analysis.
                # IMPORTANT: NO raw user text is stored - only metadata is logged.
                # Logged data: timestamp, session_id (anonymized), detection label.
                # This data is used to improve crisis detection systems.
                # ========================================================================
                detection_result = crisis_check_result.get("detection_result")
                if detection_result:
                    try:
                        log_crisis_detection(
                            session_id=session_id,
                            detection_result=detection_result,
                            use_firebase=USE_FIREBASE_LOGGING,
                            detection_method="api_chat_endpoint"
                        )
                    except Exception as log_error:
                        # Don't fail crisis detection if logging fails
                        logger.error(f"Error logging crisis detection: {log_error}")
                
                # ========================================================================
                # SAFETY HIERARCHY: Crisis detected - send crisis message and STOP
                # ========================================================================
                # When crisis is detected, we:
                # 1. Send the crisis support message (not empty response)
                # 2. Lock the conversation
                # 3. Do NOT call emotion model or response engine
                # This ensures user safety and prevents inappropriate automated responses
                # ========================================================================
                # Get crisis message in appropriate language (default messages, not translated)
                if user_wrote_urdu:
                    crisis_message_text = get_crisis_message_urdu()
                else:
                    crisis_message_text = get_crisis_message()
                
                # Clear conversation history when crisis is detected
                if session_id in conversation_history:
                    conversation_history[session_id] = []
                
                return ChatResponse(
                    responses=[BotResponse(text=crisis_message_text, sender="bot")],
                    session_id=session_id,
                    timestamp=datetime.now().isoformat(),
                    session_locked=True,
                    support_mode=True,  # Input bar should be disabled, show "Support Mode" message
                    response=crisis_message_text,  # Singular response for Flutter
                    crisis=True,  # Crisis detected
                    emotion=None  # No emotion during crisis
                )
        except Exception as e:
            # If crisis detection fails, log error but continue processing
            # We don't want technical errors to block normal conversations
            logger.error(f"Error in crisis detection: {e}")
    
    # ========================================================================
    # SAFETY HIERARCHY: No crisis detected - proceed with emotion and response
    # ========================================================================
    # Since no crisis was detected, we can safely:
    # 1. Run emotion detection model
    # 2. Generate response using LLM (to be implemented)
    # ========================================================================
    
    # Step 1: Determine crisis level (for Urdu responses context) from previous check
    if crisis_check_result:
        detection_result = crisis_check_result.get("detection_result")
        if detection_result:
            # Extract crisis level from detection result
            guardrail_severity = detection_result.get("guardrail_severity", "")
            if guardrail_severity:
                crisis_level = guardrail_severity  # "critical", "high", "medium", "low"
            elif crisis_check_result.get("crisis", False):
                crisis_level = "high"
            else:
                confidence = detection_result.get("confidence", 0.0)
                if confidence > 0.5:
                    crisis_level = "medium"
                elif confidence > 0.2:
                    crisis_level = "low"
                else:
                    crisis_level = "none"
    
    # Step 2: Run emotion detection model
    detected_emotion = None
    emotion_confidence = 0.0
    
    if emotion_predictor is not None:
        try:
            emotion_result = emotion_predictor.predict(text_for_pipeline, top_k=1)
            detected_emotion = emotion_result.get("top_emotion")
            emotion_confidence = emotion_result.get("top_confidence", 0.0)
            logger.debug(f"Emotion: {detected_emotion} ({emotion_confidence:.2f})")
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            # Continue without emotion info if emotion detection fails
    
    # Step 2: Get conversation history for this session
    # History contains up to 5 user messages + 5 assistant responses (10 messages total)
    history = conversation_history.get(session_id, [])
    logger.debug(f"History: {len(history)} messages for session {session_id}")
    
    # Step 3: Generate response using LLM (with timeout)
    # Fallback response (will be in Urdu script if user wrote Urdu, English otherwise)
    response_text = "میں یہاں ہوں، سننے کے لیے۔ آپ کیا کہنا چاہتے ہیں؟" if user_wrote_urdu else "I'm here to listen. What's on your mind?"

    logger.info("Starting LLM response generation")
    
    if llm_generator is not None:
        api_key_check = llm_generator.api_key or os.getenv("GROQ_API_KEY")
        
        # Check if API key is available
        if not api_key_check:
            logger.error("GROQ_API_KEY not available - cannot call Groq API")
        else:
            logger.info(f"Calling LLM with history length: {len(history)}")
            try:
                # SINGLE-CALL BILINGUAL GENERATION:
                # Pass original user message (Urdu if user wrote Urdu, English if English)
                # LLM generates response directly in the same language with full conversation history
                user_message_for_llm = original_urdu_message if user_wrote_urdu else text_for_pipeline
                
                try:
                    logger.info(f"Calling LLM API for session {session_id}...")
                    response_text = await asyncio.wait_for(
                        llm_generator.generate_response_async(
                            user_message=user_message_for_llm,  # Original message in user's language
                            detected_emotion=detected_emotion,
                            emotion_confidence=emotion_confidence,
                            conversation_history=history,  # History in same language as user_message
                            respond_in_urdu=user_wrote_urdu,  # LLM generates directly in Urdu if user wrote Urdu
                            original_urdu_message=original_urdu_message,  # Pass original Urdu for context
                            crisis_level=crisis_level  # Pass crisis level for context
                        ),
                        timeout=15.0  # Reduced to 15 seconds to avoid Railway gateway timeout (30s)
                    )
                    logger.info(f"LLM API call completed successfully for session {session_id}")
                    fallback_english = "I'm here to listen. What's on your mind?"
                    fallback_urdu = "میں یہاں ہوں، سننے کے لیے۔ آپ کیا کہنا چاہتے ہیں؟"
                    fallback = fallback_urdu if user_wrote_urdu else fallback_english
                    if response_text and response_text != fallback:
                        logger.info(f"LLM response generated: {len(response_text)} chars, preview: {response_text[:100]}")
                    else:
                        logger.warning("LLM returned fallback response - API call may have failed")
                except asyncio.TimeoutError:
                    logger.error(f"LLM response generation timed out after 15 seconds for session {session_id}")
                    logger.error("=" * 60)
                    # Use Urdu script fallback if user wrote Urdu
                    response_text = "میں یہاں ہوں، سننے کے لیے۔ آپ کیا کہنا چاہتے ہیں؟" if user_wrote_urdu else "I'm here to listen. What's on your mind?"
            except Exception as e:
                logger.error("=" * 60)
                logger.error(f"ERROR generating LLM response for session {session_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.error("=" * 60)
                # Use fallback response if LLM fails
                response_text = "میں یہاں ہوں، سننے کے لیے۔ آپ کیا کہنا چاہتے ہیں؟" if user_wrote_urdu else "I'm here to listen. What's on your mind?"
    else:
        logger.warning("=" * 60)
        logger.warning("WARNING: LLM generator not initialized, using fallback response")
        logger.warning("=" * 60)
    
    # Step 4: Safety net - detect if LLM returned a greeting instead of a real response
    # This catches cases where the LLM ignores pre-seeded history
    response_lower = response_text.lower().strip()
    is_greeting_response = (
        "how's your mood" in response_lower or
        "how is your mood" in response_lower or
        (response_lower.startswith("hi") and len(response_text.strip()) < 50) or
        (response_lower.startswith("hello") and len(response_text.strip()) < 50) or
        (response_lower.startswith("hey") and len(response_text.strip()) < 50)
    )
    
    if is_greeting_response and llm_generator is not None:
        logger.warning(f"LLM generated greeting despite pre-seeded history: '{response_text}' - retrying with explicit context")
        # Retry: prepend the user's message with context that greeting already happened
        retry_message = f"[IMPORTANT: You already greeted the user. Do NOT greet again. Respond to their message directly.]\n\nUser's message: {user_message_for_llm}"
        try:
            retry_response = await asyncio.wait_for(
                llm_generator.generate_response_async(
                    user_message=retry_message,
                    detected_emotion=detected_emotion,
                    emotion_confidence=emotion_confidence,
                    conversation_history=history,
                    respond_in_urdu=user_wrote_urdu,
                    original_urdu_message=original_urdu_message,
                    crisis_level=crisis_level
                ),
                timeout=10.0
            )
            retry_lower = (retry_response or "").lower().strip()
            retry_is_greeting = (
                "how's your mood" in retry_lower or
                "how is your mood" in retry_lower or
                (retry_lower.startswith("hi") and len(retry_lower) < 50) or
                (retry_lower.startswith("hello") and len(retry_lower) < 50)
            )
            if retry_response and not retry_is_greeting:
                response_text = retry_response
                logger.info(f"Retry succeeded with proper response: {response_text[:100]}")
            else:
                logger.warning(f"Retry also returned greeting: '{retry_response}' - using fallback")
                response_text = "میں یہاں ہوں، سننے کے لیے۔ آپ کیا کہنا چاہتے ہیں؟" if user_wrote_urdu else "I'm here to listen. What's on your mind?"
        except Exception as e:
            logger.error(f"Retry failed: {e}")
            response_text = "میں یہاں ہوں، سننے کے لیے۔ آپ کیا کہنا چاہتے ہیں؟" if user_wrote_urdu else "I'm here to listen. What's on your mind?"
    
    # Step 5: Use LLM response directly (already in Urdu if user wrote Urdu, English otherwise)
    # No translation needed - LLM generates directly in the target language
    response_for_user = response_text
    
    # Step 6: Update conversation history
    # SINGLE-CALL BILINGUAL GENERATION: Store messages in their original language
    # This ensures conversation history is in the same language for context continuity
    # IMPORTANT: For Urdu conversations, bot responses are always in proper Urdu script (not Roman Urdu)
    # User messages may be in Roman Urdu or proper Urdu script - LLM can understand both
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Add user message in original language (Urdu if user wrote Urdu, English if English)
    # This ensures history is in the same language for proper context
    # Note: User messages may be in Roman Urdu (e.g., "mai pareshan hu") - that's fine, LLM understands it
    user_message_for_history = original_urdu_message if user_wrote_urdu else text_for_pipeline
    conversation_history[session_id].append({
        "role": "user",
        "content": user_message_for_history  # Original language (Urdu or English)
    })
    
    # Add bot response (in same language as user message - Urdu if user wrote Urdu, English otherwise)
    # IMPORTANT: Bot responses in Urdu are ALWAYS in proper Urdu script (مجھے افسوس ہے), not Roman Urdu
    # This maintains language consistency and proper Urdu script throughout the conversation
    conversation_history[session_id].append({
        "role": "assistant",
        "content": response_text  # Already in correct language (proper Urdu script or English)
    })
    
    # Keep only last MAX_HISTORY_LENGTH messages (trim from the beginning)
    # This stores 5 user messages + 5 assistant responses = 10 messages total
    if len(conversation_history[session_id]) > MAX_HISTORY_LENGTH:
        conversation_history[session_id] = conversation_history[session_id][-MAX_HISTORY_LENGTH:]
        logger.info(f"Trimmed conversation history to last {MAX_HISTORY_LENGTH} messages (5 user + 5 assistant)")
    
    # Format response for Flutter app (use translated response if user wrote Urdu)
    bot_responses = [BotResponse(text=response_for_user, sender="bot")]
    
    # Return formatted response (session is not locked, so input bar should be enabled)
    logger.info(f"Returning response to Flutter app: {len(response_for_user)} chars")
    return ChatResponse(
        responses=bot_responses,
        session_id=session_id,
        timestamp=datetime.now().isoformat(),
        session_locked=False,
        support_mode=False,  # Input bar should be enabled
        response=response_for_user,  # Singular response for Flutter (Urdu if user wrote Urdu)
        crisis=False,  # No crisis detected
        emotion=detected_emotion  # Detected emotion
    )


@app.post("/chat/message")
async def chat_simple(message: ChatMessage):
    """
    Simplified chat endpoint that returns just the response text.
    
    This is an alternative endpoint for simpler integration.
    Returns only the first response text as a string.
    
    SAFETY GUARDRAIL: This endpoint also respects conversation locking.
    If a crisis is detected or conversation is locked, returns empty response.
    
    Args:
        message: ChatMessage object
    
    Returns:
        Dictionary with response text and session_id
    """
    logger.info("=" * 80)
    logger.info("CHAT/MESSAGE ENDPOINT CALLED")
    logger.info("=" * 80)
    # Use the main chat endpoint (which includes safety guardrails)
    full_response = await chat(message)
    
    # Extract first response text (will be empty if conversation is locked)
    response_text = full_response.responses[0].text if full_response.responses else ""
    
    return {
        "response": response_text,
        "session_id": full_response.session_id,
        "session_locked": full_response.session_locked,
        "support_mode": full_response.support_mode
    }


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """
    Get conversation history for a session.
    
    Note: This is a placeholder. In production, you would need to:
    1. Store conversation history in a database
    2. Retrieve history for the given session_id
    3. Return formatted conversation history
    
    For now, this returns a message indicating history is not implemented.
    
    Args:
        session_id: The session identifier
    
    Returns:
        Message indicating feature is not yet implemented
    """
    return {
        "message": "Conversation history is not yet implemented",
        "session_id": session_id,
        "note": "This feature requires database integration to store and retrieve conversation history"
    }


@app.post("/personalize-content", response_model=PersonalizeContentResponse)
async def personalize_content(request: PersonalizeContentRequest):
    """
    AI-powered content personalization endpoint.
    
    Selects personalized content (3 lessons + 1 activity) based on:
    - User's current mood (0-4)
    - User's selected focus areas
    - Recently shown content (to avoid repetition)
    - Available content metadata
    
    Uses LLM to intelligently select content that matches the user's current emotional state
    and learning needs.
    
    Args:
        request: PersonalizeContentRequest with mood, focus areas, and content metadata
    
    Returns:
        PersonalizeContentResponse with selected lesson IDs and activity ID
    
    Raises:
        HTTPException: If request is invalid or LLM is unavailable
    """
    logger.info("=" * 80)
    logger.info("PERSONALIZE-CONTENT ENDPOINT CALLED")
    logger.info(f"Mood: {request.mood}, Focus areas: {request.focus_areas}")
    logger.info(f"Available content: {len(request.available_content)} items")
    logger.info("=" * 80)
    
    # Validate request
    if not request.available_content:
        logger.warning("No available content provided, returning empty selection")
        return PersonalizeContentResponse(
            selected_lessons=[],
            selected_activity=None
        )
    
    # Filter out recently shown content, but keep some buffer to avoid filtering everything
    recently_shown_ids = set(request.recently_shown_lessons + request.recently_shown_activities)
    available_content = [
        item for item in request.available_content
        if item.get('id') not in recently_shown_ids
    ]
    
    # Check how many lessons vs activities we have after filtering
    filtered_lessons = [item for item in available_content if item.get('type') == 'lesson']
    filtered_activities = [item for item in available_content if item.get('type') == 'activity']
    
    logger.info(f"After filtering recently shown: {len(filtered_lessons)} lessons, {len(filtered_activities)} activities")
    
    # If we have very few lessons (need at least 3), allow some recently shown lessons
    # This prevents the system from getting stuck when most lessons have been shown
    if len(filtered_lessons) < 3:
        logger.warning(f"Only {len(filtered_lessons)} lessons available after filtering. Allowing some recently shown lessons to ensure selection is possible.")
        # Re-add recently shown lessons (but keep activities filtered)
        available_content = [
            item for item in request.available_content
            if item.get('id') not in request.recently_shown_activities
            or item.get('type') == 'lesson'  # Allow recently shown lessons
        ]
        logger.info(f"Expanded to allow recently shown lessons. Now have {len([i for i in available_content if i.get('type') == 'lesson'])} lessons")
    
    # If we have no activities, allow recently shown activities (we need at least 1)
    if len(filtered_activities) == 0:
        logger.warning(f"No activities available after filtering. Allowing recently shown activities to ensure we have at least one.")
        # Re-add recently shown activities
        available_content = [
            item for item in request.available_content
            if item.get('id') not in request.recently_shown_lessons
            or item.get('type') == 'activity'  # Allow recently shown activities
        ]
        logger.info(f"Expanded to allow recently shown activities. Now have {len([i for i in available_content if i.get('type') == 'activity'])} activities")
    
    # If filtering removed too much content overall, allow some recently shown content
    if len(available_content) < 5:
        logger.warning(f"Only {len(available_content)} items total available after filtering. Allowing some recently shown content to ensure variety.")
        available_content = request.available_content
        logger.info(f"Expanded to {len(available_content)} items to ensure selection is possible")
    
    if not available_content:
        logger.warning("No available content, returning empty selection")
        return PersonalizeContentResponse(
            selected_lessons=[],
            selected_activity=None
        )
    
    # Separate lessons and activities
    available_lessons = [item for item in available_content if item.get('type') == 'lesson']
    available_activities = [item for item in available_content if item.get('type') == 'activity']
    
    logger.info(f"After filtering: {len(available_lessons)} lessons, {len(available_activities)} activities")
    
    # Check if we have enough content from focus areas
    focus_area_lessons = [
        lesson for lesson in available_lessons
        if lesson.get('focusArea') in request.focus_areas
    ]
    focus_area_activities = [
        act for act in available_activities
        if act.get('focusArea') in request.focus_areas
    ]
    
    # Determine if we need to expand beyond focus areas
    # If we have less than 3 lessons or no activities from focus areas, allow other areas
    use_other_areas = len(focus_area_lessons) < 3 or len(focus_area_activities) == 0
    
    if use_other_areas:
        logger.info(f"Not enough content from focus areas ({len(focus_area_lessons)} lessons, {len(focus_area_activities)} activities). Expanding to other areas to avoid repetition.")
    else:
        logger.info(f"Sufficient content from focus areas ({len(focus_area_lessons)} lessons, {len(focus_area_activities)} activities). Using only focus areas.")
    
    # Try to use LLM for intelligent selection
    if llm_generator is not None and llm_generator.api_key:
        try:
            # If we have enough from focus areas, prioritize them; otherwise use all available
            lessons_to_use = focus_area_lessons if not use_other_areas else available_lessons
            activities_to_use = focus_area_activities if not use_other_areas else available_activities
            
            selected_lessons, selected_activity = await _select_content_with_llm(
                llm_generator=llm_generator,
                mood=request.mood,
                focus_areas=request.focus_areas,
                available_lessons=lessons_to_use,
                available_activities=activities_to_use,
                prefer_focus_areas=not use_other_areas,
                recently_shown_lessons=request.recently_shown_lessons,
                recently_shown_activities=request.recently_shown_activities
            )
            
            # CRITICAL: Ensure we have exactly 3 lessons from STRICTLY 3 different focus areas
            # This is a hard requirement - no exceptions
            if len(selected_lessons) < 3:
                logger.warning(f"LLM returned only {len(selected_lessons)} valid lessons, filling to 3 with STRICT diversity...")
                # Get focus areas already used
                used_areas = set()
                for lid in selected_lessons:
                    lesson = next((l for l in available_lessons if l['id'] == lid), None)
                    if lesson:
                        used_areas.add(lesson.get('focusArea'))
                
                # Group ALL remaining lessons by focus area (from all available, not just focus areas)
                remaining_by_area = {}
                for item in available_lessons:
                    if item['id'] not in selected_lessons:
                        area = item.get('focusArea')
                        if area not in remaining_by_area:
                            remaining_by_area[area] = []
                        remaining_by_area[area].append(item)
                
                logger.info(f"Filling lessons: {len(selected_lessons)} selected, {len(remaining_by_area)} focus areas available")
                logger.info(f"Already used areas: {used_areas}")
                
                # Strategy: Get ONE lesson from EACH different focus area
                # Step 1: Try to get one from each of user's focus areas (if not already used)
                priority_areas = [area for area in request.focus_areas if area not in used_areas and area in remaining_by_area]
                
                # Select one from each priority area
                for area in priority_areas:
                    if len(selected_lessons) >= 3:
                        break
                    if area in remaining_by_area and remaining_by_area[area]:
                        selected_lessons.append(remaining_by_area[area][0]['id'])
                        used_areas.add(area)
                        logger.info(f"Added lesson from user's focus area: {area}")
                
                # Step 2: If still need more, get from OTHER topics (not user's focus areas)
                if len(selected_lessons) < 3:
                    other_areas = [area for area in remaining_by_area.keys() 
                                  if area not in used_areas and area not in request.focus_areas]
                    for area in other_areas:
                        if len(selected_lessons) >= 3:
                            break
                        if area in remaining_by_area and remaining_by_area[area]:
                            selected_lessons.append(remaining_by_area[area][0]['id'])
                            used_areas.add(area)
                            logger.info(f"Added lesson from other topic: {area}")
                
                # Step 3: If STILL need more (all topics exhausted), allow repetition but maintain diversity
                # Try to get from any unused area (including focus areas we already used, but different ones)
                if len(selected_lessons) < 3:
                    available_areas = [area for area in remaining_by_area.keys() if area not in used_areas]
                    for area in available_areas:
                        if len(selected_lessons) >= 3:
                            break
                        if area in remaining_by_area and remaining_by_area[area]:
                            selected_lessons.append(remaining_by_area[area][0]['id'])
                            used_areas.add(area)
                            logger.info(f"Added lesson from unused area (repetition allowed): {area}")
                
                # Step 4: Last resort - if we still don't have 3, fill from any available
                # This should rarely happen, but ensures we always return 3 lessons
                if len(selected_lessons) < 3:
                    logger.warning(f"CRITICAL: Could only fill to {len(selected_lessons)} lessons from different areas. Filling remaining slots...")
                    for item in available_lessons:
                        if len(selected_lessons) >= 3:
                            break
                        if item['id'] not in selected_lessons:
                            selected_lessons.append(item['id'])
                            area = item.get('focusArea')
                            if area not in used_areas:
                                used_areas.add(area)
                                logger.info(f"Added lesson {item['id']} from area {area} as last resort")
                
                # Final validation: ensure we have 3 different areas
                final_areas = [next((l.get('focusArea') for l in available_lessons if l['id'] == lid), 'unknown') for lid in selected_lessons]
                unique_final_areas = set(final_areas)
                logger.info(f"Final selection: {len(selected_lessons)} lessons from {len(unique_final_areas)} unique areas: {final_areas}")
                
                if len(selected_lessons) < 3:
                    logger.warning(f"CRITICAL: Could only fill to {len(selected_lessons)} lessons (need 3)")
                elif len(unique_final_areas) < 3:
                    logger.error(f"CRITICAL WARNING: Have {len(selected_lessons)} lessons but only {len(unique_final_areas)} unique areas: {final_areas}")
            
            # Ensure we have 1 activity (use fallback if needed)
            if not selected_activity:
                logger.warning("LLM didn't select an activity, using fallback...")
                # First try focus areas
                focus_area_activity_ids = [
                    item['id'] for item in available_activities
                    if item.get('focusArea') in request.focus_areas
                ]
                if focus_area_activity_ids:
                    selected_activity = focus_area_activity_ids[0]
                    logger.info(f"Selected activity from focus area: {selected_activity}")
                # If no focus area activity and we're allowing other areas, use any
                elif use_other_areas and available_activities:
                    selected_activity = available_activities[0].get('id')
                    logger.info(f"Selected activity from other area: {selected_activity}")
                # Last resort: If still no activity, log warning
                if not selected_activity:
                    logger.error(f"CRITICAL: No activity available! Total activities: {len(available_activities)}")
                    logger.error("This means all activities were filtered out. Check activity filtering logic.")
            
            logger.info(f"LLM selected: {len(selected_lessons)} lessons, activity: {selected_activity}")
            
            # If LLM returned 0 lessons, fall back to rule-based selection immediately
            if len(selected_lessons) == 0:
                logger.warning("=" * 60)
                logger.warning("LLM returned 0 lessons after validation!")
                logger.warning("This usually means LLM selected lessons that don't match focus areas or are invalid")
                logger.warning("FALLING BACK TO RULE-BASED SELECTION...")
                logger.warning("=" * 60)
                # Fall through to rule-based selection below - don't return here
            elif len(selected_lessons) < 3:
                # If we have some lessons but not enough, try to fill them first
                logger.info(f"LLM returned {len(selected_lessons)} lessons, attempting to fill to 3...")
                # The filling logic above should have already run, so if we still have < 3, return what we have
                # or fall back if we have 0
                if len(selected_lessons) == 0:
                    logger.warning("Could not fill any lessons, falling back to rule-based selection")
                    # Fall through to rule-based selection
                else:
                    return PersonalizeContentResponse(
                        selected_lessons=selected_lessons[:3],  # Ensure max 3
                        selected_activity=selected_activity
                    )
            else:
                # We have 3 or more lessons, return them
                return PersonalizeContentResponse(
                    selected_lessons=selected_lessons[:3],  # Ensure max 3
                    selected_activity=selected_activity
                )
        except Exception as e:
            logger.error(f"LLM selection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fall through to rule-based selection
    else:
        logger.warning("LLM not available, using rule-based selection")
    
    # Fallback: Rule-based selection
    logger.info("=" * 60)
    logger.info("USING RULE-BASED CONTENT SELECTION")
    logger.info(f"Available: {len(available_lessons)} lessons, {len(available_activities)} activities")
    logger.info("=" * 60)
    # First try focus areas, expand to other areas if not enough content
    selected_lessons = []
    selected_activity = None
    
    # Determine which lessons/activities to use based on availability
    lessons_to_use = focus_area_lessons if not use_other_areas else available_lessons
    activities_to_use = focus_area_activities if not use_other_areas else available_activities
    
    logger.info(f"Using {len(lessons_to_use)} lessons and {len(activities_to_use)} activities for selection")
    
    # Select 3 lessons - ONE from EACH different focus area
    if lessons_to_use:
        # Group lessons by focus area
        lessons_by_area = {}
        for lesson in lessons_to_use:
            area = lesson.get('focusArea', 'unknown')
            if area not in lessons_by_area:
                lessons_by_area[area] = []
            lessons_by_area[area].append(lesson)
        
        # Sort by mood suitability (simple heuristic)
        mood_keywords = {
            0: ['support', 'gentle', 'self-compassion', 'breathing'],
            1: ['support', 'gentle', 'mindfulness', 'calm'],
            2: ['balanced', 'understanding', 'awareness'],
            3: ['growth', 'building', 'skills', 'challenge'],
            4: ['growth', 'building', 'advanced', 'challenge']
        }
        
        keywords = mood_keywords.get(request.mood, [])
        
        # Score and rank lessons within each focus area
        ranked_lessons_by_area = {}
        for area, area_lessons in lessons_by_area.items():
            scored_lessons = []
            for lesson in area_lessons:
                score = 0
                title = lesson.get('title', '').lower()
                content = lesson.get('content', '').lower()
                text = title + ' ' + content
                
                for keyword in keywords:
                    if keyword in text:
                        score += 1
                
                # Prioritize lessons from user's focus areas (higher score)
                if area in request.focus_areas:
                    score += 10  # Strong preference for focus areas
                
                scored_lessons.append((score, lesson))
            
            # Sort by score (descending) and store
            scored_lessons.sort(key=lambda x: x[0], reverse=True)
            ranked_lessons_by_area[area] = [lesson for _, lesson in scored_lessons]
        
        # CRITICAL: Select STRICTLY one lesson from each different focus area (3 different topics)
        selected_lessons = []
        used_areas = set()
        
        # Strategy: Get ONE lesson from EACH different focus area
        # Step 1: First pass - Get one from each user's focus area (prioritize user's focus areas)
        for focus_area in request.focus_areas:
            if len(selected_lessons) >= 3:
                break
            if focus_area in ranked_lessons_by_area and focus_area not in used_areas:
                area_lessons = ranked_lessons_by_area[focus_area]
                if area_lessons:
                    selected_lessons.append(area_lessons[0]['id'])
                    used_areas.add(focus_area)
                    logger.info(f"Rule-based: Selected lesson from user's focus area: {focus_area}")
        
        # Step 2: If we need more, get from OTHER topics (not user's focus areas)
        # This ensures variety even when user has fewer than 3 focus areas
        if len(selected_lessons) < 3:
            # Get all available areas that are NOT user's focus areas
            other_areas = [
                area for area in ranked_lessons_by_area.keys()
                if area not in used_areas and area not in request.focus_areas
            ]
            
            # Get one from each other area
            for area in other_areas:
                if len(selected_lessons) >= 3:
                    break
                if area in ranked_lessons_by_area:
                    area_lessons = ranked_lessons_by_area[area]
                    if area_lessons:
                        selected_lessons.append(area_lessons[0]['id'])
                        used_areas.add(area)
                        logger.info(f"Rule-based: Selected lesson from other topic: {area}")
        
        # Step 3: If STILL need more (all topics exhausted), allow repetition but maintain diversity
        # Try to get from any unused area (including focus areas we already used, but different ones)
        if len(selected_lessons) < 3:
            available_areas = [
                area for area in ranked_lessons_by_area.keys()
                if area not in used_areas
            ]
            
            for area in available_areas:
                if len(selected_lessons) >= 3:
                    break
                if area in ranked_lessons_by_area:
                    area_lessons = ranked_lessons_by_area[area]
                    if area_lessons:
                        selected_lessons.append(area_lessons[0]['id'])
                        used_areas.add(area)
                        logger.info(f"Rule-based: Selected lesson from unused area (repetition allowed): {area}")
        
        # Step 4: Last resort - if we still don't have 3, fill from any available
        # This should rarely happen, but ensures we always return 3 lessons
        if len(selected_lessons) < 3:
            logger.warning(f"CRITICAL: Only {len(selected_lessons)} lessons selected. Filling from any available...")
            for area in ranked_lessons_by_area.keys():
                if len(selected_lessons) >= 3:
                    break
                area_lessons = ranked_lessons_by_area[area]
                for lesson in area_lessons:
                    if len(selected_lessons) >= 3:
                        break
                    if lesson['id'] not in selected_lessons:
                        selected_lessons.append(lesson['id'])
                        if area not in used_areas:
                            used_areas.add(area)
        
        # Log which focus areas were selected
        if selected_lessons:
            selected_focus_areas = [
                next((l.get('focusArea') for l in lessons_to_use if l['id'] == lid), 'unknown')
                for lid in selected_lessons
            ]
            focus_area_count = sum(1 for fa in selected_focus_areas if fa in request.focus_areas)
            logger.info(f"Rule-based selected {len(selected_lessons)} lessons from {len(set(selected_focus_areas))} different focus areas")
            logger.info(f"Selected focus areas: {selected_focus_areas}")
            logger.info(f"Focus area diversity: {len(set(selected_focus_areas))} unique areas")
    
    # Select 1 activity based on mood (prefer focus areas, but use others if needed)
    if activities_to_use:
        # Score activities: prioritize focus areas and match mood
        scored_activities = []
        for act in activities_to_use:
            score = 0
            
            # Strong preference for focus areas
            if act.get('focusArea') in request.focus_areas:
                score += 10
            
            # Match difficulty to mood
            difficulty = act.get('difficulty', '').lower()
            if request.mood <= 1 and difficulty == 'easy':
                score += 5
            elif request.mood >= 3 and difficulty == 'hard':
                score += 5
            elif request.mood == 2:  # Medium mood - any difficulty
                score += 2
            
            scored_activities.append((score, act))
        
        # Sort by score and select best
        scored_activities.sort(key=lambda x: x[0], reverse=True)
        if scored_activities:
            selected_activity = scored_activities[0][1].get('id')
            activity_focus = scored_activities[0][1].get('focusArea', 'unknown')
            is_from_focus = activity_focus in request.focus_areas
            logger.info(f"Rule-based selected activity from {'focus area' if is_from_focus else 'other area'}: {activity_focus}")
    
    logger.info(f"Rule-based selected: {len(selected_lessons)} lessons, activity: {selected_activity}")
    return PersonalizeContentResponse(
        selected_lessons=selected_lessons[:3],
        selected_activity=selected_activity
    )


async def _select_content_with_llm(
    llm_generator: LLMResponseGenerator,
    mood: int,
    focus_areas: List[str],
    available_lessons: List[Dict[str, Any]],
    available_activities: List[Dict[str, Any]],
    prefer_focus_areas: bool = True,
    recently_shown_lessons: Optional[List[str]] = None,
    recently_shown_activities: Optional[List[str]] = None
) -> Tuple[List[str], Optional[str]]:
    """
    Use LLM to intelligently select content based on mood and context.
    
    Args:
        llm_generator: LLM response generator instance
        mood: Current mood (0-4)
        focus_areas: User's focus areas
        available_lessons: List of available lesson metadata
        available_activities: List of available activity metadata
        prefer_focus_areas: Whether to strictly use only focus areas
        recently_shown_lessons: List of recently shown lesson IDs to avoid
        recently_shown_activities: List of recently shown activity IDs to avoid
    
    Returns:
        Tuple of (selected_lesson_ids, selected_activity_id)
    """
    # Build prompt for content selection
    mood_descriptions = {
        0: "very sad, low energy, needs gentle support",
        1: "sad, low mood, needs supportive content",
        2: "neutral, balanced mood, can handle various content",
        3: "good mood, positive energy, can handle challenging content",
        4: "very happy, high energy, ready for growth-oriented content"
    }
    
    mood_desc = mood_descriptions.get(mood, "neutral mood")
    
    # Log what we're sending to LLM
    logger.info(f"LLM selection: {len(available_lessons)} lessons, {len(available_activities)} activities available")
    logger.info(f"Focus areas: {focus_areas}, Prefer focus areas: {prefer_focus_areas}")
    if not available_lessons:
        logger.error("No lessons available for LLM selection - this should not happen!")
        raise Exception("No lessons available for LLM selection")
    
    # Format recently shown content for the prompt
    recently_shown_lessons_list = recently_shown_lessons or []
    recently_shown_activities_list = recently_shown_activities or []
    
    recently_shown_text = ""
    if recently_shown_lessons_list or recently_shown_activities_list:
        recently_shown_text = f"""
IMPORTANT - AVOID REPETITION: The following content was recently shown to this user and should NOT be selected:
- Recently shown lessons: {', '.join(recently_shown_lessons_list[:10]) if recently_shown_lessons_list else 'None'}
- Recently shown activities: {', '.join(recently_shown_activities_list[:10]) if recently_shown_activities_list else 'None'}

CRITICAL: You MUST select DIFFERENT content that has NOT been shown recently. This ensures variety and prevents the user from seeing the same content repeatedly.
"""
    
    # Format available content for the prompt
    # Limit to reasonable number to avoid timeout and token limits
    # Show more options if we have fewer items, but cap at reasonable limit
    max_lessons_to_show = min(25, len(available_lessons))  # Show up to 25 lessons
    max_activities_to_show = min(15, len(available_activities))  # Show up to 15 activities
    
    lessons_text = "\n".join([
        f"- ID: {lesson['id']}, Title: {lesson.get('title', 'N/A')}, "
        f"Focus: {lesson.get('focusArea', 'N/A')}"
        # Removed preview to reduce token count and speed up response
        for lesson in available_lessons[:max_lessons_to_show]
    ])
    
    activities_text = "\n".join([
        f"- ID: {act['id']}, Title: {act.get('title', 'N/A')}, "
        f"Difficulty: {act.get('difficulty', 'N/A')}, "
        f"Focus: {act.get('focusArea', 'N/A')}"
        for act in available_activities[:max_activities_to_show]
    ])
    
    logger.info(f"Formatted {max_lessons_to_show} lessons and {max_activities_to_show} activities for LLM prompt")
    
    # Build prompt based on whether we're strictly using focus areas or allowing others
    if prefer_focus_areas:
        focus_area_instruction = f"""
CRITICAL REQUIREMENT: ALL selected content MUST come from the user's selected focus areas: {', '.join(focus_areas)}
- Every lesson ID you select MUST have a "Focus" field matching one of these focus areas
- The activity ID you select MUST have a "Focus" field matching one of these focus areas
- Do NOT select any content that does not belong to these focus areas"""
    else:
        focus_area_instruction = f"""
PREFERENCE: Strongly prefer content from the user's selected focus areas: {', '.join(focus_areas)}
- Prioritize lessons and activities with "Focus" field matching these focus areas
- Only select content from other focus areas if there is insufficient content from the user's focus areas
- This ensures variety and avoids repetition when the user has completed most content from their focus areas"""
    
    system_prompt = f"""You are a mental health content personalization assistant.

Your task is to select the most appropriate content for a user based on their current mood and focus areas.
{focus_area_instruction}
{recently_shown_text}

User's current mood: {mood} ({mood_desc})
User's selected focus areas: {', '.join(focus_areas)}

Available Lessons:
{lessons_text if lessons_text else "No lessons available"}

Available Activities:
{activities_text if activities_text else "No activities available"}

CRITICAL SELECTION REQUIREMENTS:

1. VARIETY IS ESSENTIAL: Select DIFFERENT content each time. Do NOT select the same lessons or activities that were recently shown.

2. LESSON SELECTION - STRICT DIVERSITY REQUIREMENT: Select exactly 3 lessons, with ONE lesson from EACH DIFFERENT focus area.
   - THIS IS MANDATORY: You MUST select 3 lessons from 3 DIFFERENT focus areas. NO EXCEPTIONS.
   - If the user has 3 or more focus areas, select one lesson from each of 3 DIFFERENT focus areas
   - If the user has fewer than 3 focus areas, select one from each focus area, then select from OTHER available focus areas (not user's focus areas) to maintain diversity
   - DO NOT select multiple lessons from the same focus area - this will be rejected
   - If all user's focus areas have been covered, use OTHER topics (recommended), and only when all topics have been covered, then start repeating them
   - This ensures variety and covers different aspects of the user's mental health journey
   - Example: If user has focus areas [anxiety, depression], select one from anxiety, one from depression, and one from another topic like self-esteem or emotional-regulation

3. ACTIVITY SELECTION: Select exactly 1 activity that:
   - Matches their mood and energy level
   - Is DIFFERENT from recently shown activities
   - Provides variety in their daily routine

4. MOOD-BASED SELECTION:
   - For low mood (0-1): Choose gentle, supportive lessons and easy activities
   - For neutral mood (2): Choose balanced, educational content
   - For high mood (3-4): Choose growth-oriented, challenging content

5. VALIDATION:
   - Verify each selected lesson ID exists in the available lessons list above
   - Verify the selected activity ID exists in the available activities list above
   - If prefer_focus_areas is true, verify each selected ID's "Focus" field matches one of: {', '.join(focus_areas)}
   - CRITICAL: Verify that all 3 selected lessons come from 3 DIFFERENT focus areas - check the "Focus" field of each lesson
   - If you select multiple lessons from the same focus area, your selection will be automatically corrected, but this is not ideal

Respond ONLY with a JSON object in this exact format:
{{
  "selected_lessons": ["lesson_id_1", "lesson_id_2", "lesson_id_3"],
  "selected_activity": "activity_id"
}}

Do not include any other text, explanations, or markdown formatting. Only return the JSON object."""

    try:
        # Call LLM
        response_text = await asyncio.wait_for(
            llm_generator._call_groq_api_async(
                system_prompt=system_prompt,
                user_message="Please select personalized content for this user.",
                detected_emotion=None,
                emotion_confidence=0.0,
                conversation_history=None
            ),
            timeout=35.0  # Increased timeout to match Flutter's 45s (with buffer)
        )
        
        if not response_text:
            raise Exception("LLM returned empty response")
        
        # Parse JSON response
        import json
        # Try to extract JSON from response (in case LLM adds extra text)
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        data = json.loads(response_text)
        
        selected_lessons = data.get('selected_lessons', [])
        selected_activity = data.get('selected_activity')
        
        # Validate lesson IDs exist
        valid_lesson_ids = {lesson['id'] for lesson in available_lessons}
        # Create a map of lesson ID to focus area for validation
        lesson_focus_map = {lesson['id']: lesson.get('focusArea') for lesson in available_lessons}
        
        # Filter to valid lessons, and validate focus areas if required
        validated_lessons = []
        invalid_lessons = []
        logger.info(f"Validating {len(selected_lessons)} lessons from LLM...")
        logger.info(f"Available lesson IDs count: {len(valid_lesson_ids)}")
        logger.info(f"Prefer focus areas: {prefer_focus_areas}, Focus areas: {focus_areas}")
        
        for lid in selected_lessons:
            if lid not in valid_lesson_ids:
                invalid_lessons.append(f"{lid} (not in available lessons)")
                logger.warning(f"LLM selected lesson ID '{lid}' which is not in available lessons")
                continue
            lesson_focus = lesson_focus_map.get(lid)
            logger.info(f"Lesson {lid} has focus area: {lesson_focus}")
            if prefer_focus_areas:
                # Strict: only allow focus areas
                if lesson_focus and lesson_focus in focus_areas:
                    validated_lessons.append(lid)
                    logger.info(f"Lesson {lid} validated (focus area matches)")
                else:
                    invalid_lessons.append(f"{lid} (focus area '{lesson_focus}' not in {focus_areas})")
                    logger.warning(f"LLM selected lesson {lid} from focus area '{lesson_focus}' which is not in user's focus areas {focus_areas}")
            else:
                # Flexible: allow all, but log which are from focus areas
                validated_lessons.append(lid)
                logger.info(f"Lesson {lid} validated (flexible mode)")
        
        if invalid_lessons:
            logger.warning(f"LLM selected {len(invalid_lessons)} invalid lessons: {invalid_lessons}")
        if len(validated_lessons) == 0 and len(selected_lessons) > 0:
            logger.error(f"LLM selected {len(selected_lessons)} lessons but ALL were invalid after validation!")
            logger.error(f"Sample available lesson IDs: {list(valid_lesson_ids)[:5]}")
            logger.error(f"LLM selected IDs: {selected_lessons}")
            logger.error(f"Available lessons focus areas: {set(lesson_focus_map.values())}")
        
        selected_lessons = validated_lessons
        logger.info(f"After validation: {len(selected_lessons)} valid lessons")
        
        # CRITICAL: Ensure lessons come from STRICTLY 3 different focus areas
        # This is a hard requirement - no exceptions
        if len(selected_lessons) > 0:
            selected_focus_areas_list = [lesson_focus_map.get(lid, 'unknown') for lid in selected_lessons]
            unique_areas = set(selected_focus_areas_list)
            
            # If we have duplicates (same focus area), fix it STRICTLY
            if len(unique_areas) < len(selected_lessons) or len(selected_lessons) < 3:
                logger.warning(f"LLM selection doesn't meet diversity requirement. Ensuring STRICT diversity...")
                logger.warning(f"Original selection: {selected_focus_areas_list} (unique: {len(unique_areas)})")
                
                # Re-select to STRICTLY ensure 3 different focus areas
                diverse_lessons = []
                used_areas_for_diversity = set()
                
                # Strategy: Get ONE lesson from EACH different focus area
                # Step 1: Group all available lessons by focus area
                lessons_by_area = {}
                for lesson_item in available_lessons:
                    area = lesson_item.get('focusArea', 'unknown')
                    if area not in lessons_by_area:
                        lessons_by_area[area] = []
                    lessons_by_area[area].append(lesson_item)
                
                # Step 2: First, try to get one from each user's focus areas (if not already used)
                for focus_area in focus_areas:
                    if len(diverse_lessons) >= 3:
                        break
                    if focus_area in lessons_by_area and focus_area not in used_areas_for_diversity:
                        # Get first available lesson from this focus area
                        for lesson_item in lessons_by_area[focus_area]:
                            lesson_id = lesson_item.get('id')
                            if lesson_id not in diverse_lessons:
                                diverse_lessons.append(lesson_id)
                                used_areas_for_diversity.add(focus_area)
                                break
                
                # Step 3: If still need more, get from OTHER topics (not user's focus areas)
                if len(diverse_lessons) < 3:
                    for area in lessons_by_area.keys():
                        if len(diverse_lessons) >= 3:
                            break
                        # Skip if already used or if it's a user focus area (we already tried those)
                        if area not in used_areas_for_diversity and area not in focus_areas:
                            for lesson_item in lessons_by_area[area]:
                                lesson_id = lesson_item.get('id')
                                if lesson_id not in diverse_lessons:
                                    diverse_lessons.append(lesson_id)
                                    used_areas_for_diversity.add(area)
                                    break
                
                # Step 4: If STILL need more (all topics exhausted), allow repetition but maintain diversity
                # This means we can repeat focus areas, but still ensure 3 different ones
                if len(diverse_lessons) < 3:
                    # Try to get from any unused area (including focus areas we already used)
                    for area in lessons_by_area.keys():
                        if len(diverse_lessons) >= 3:
                            break
                        if area not in used_areas_for_diversity:
                            for lesson_item in lessons_by_area[area]:
                                lesson_id = lesson_item.get('id')
                                if lesson_id not in diverse_lessons:
                                    diverse_lessons.append(lesson_id)
                                    used_areas_for_diversity.add(area)
                                    break
                
                # Step 5: Last resort - if we still don't have 3 different areas, 
                # we'll allow same area but log a critical warning
                if len(diverse_lessons) < 3:
                    logger.error(f"CRITICAL: Could only get {len(diverse_lessons)} lessons from {len(used_areas_for_diversity)} different areas. Available areas: {list(lessons_by_area.keys())}")
                    # Fill remaining slots from any available lessons
                    for lesson_item in available_lessons:
                        if len(diverse_lessons) >= 3:
                            break
                        lesson_id = lesson_item.get('id')
                        if lesson_id not in diverse_lessons:
                            diverse_lessons.append(lesson_id)
                            area = lesson_item.get('focusArea', 'unknown')
                            if area not in used_areas_for_diversity:
                                used_areas_for_diversity.add(area)
                
                selected_lessons = diverse_lessons[:3]
                final_areas = [lesson_focus_map.get(lid, 'unknown') for lid in selected_lessons]
                logger.info(f"STRICT diversity enforced: {len(set(final_areas))} unique areas from {final_areas}")
                
                # Final validation: ensure we have 3 different areas
                if len(set(final_areas)) < 3 and len(selected_lessons) == 3:
                    logger.error(f"CRITICAL WARNING: Still have duplicates! Areas: {final_areas}")
        
        # Validate activity ID exists
        valid_activity_ids = {act['id'] for act in available_activities}
        activity_focus_map = {act['id']: act.get('focusArea') for act in available_activities}
        
        if selected_activity:
            if selected_activity in valid_activity_ids:
                activity_focus = activity_focus_map.get(selected_activity)
                if prefer_focus_areas:
                    # Strict: only allow focus areas
                    if not activity_focus or activity_focus not in focus_areas:
                        logger.warning(f"LLM selected activity {selected_activity} from focus area '{activity_focus}' which is not in user's focus areas {focus_areas}")
                        selected_activity = None
                # If not strict, allow any valid activity
            else:
                selected_activity = None
        
        # Log which focus areas the selected content comes from
        if selected_lessons:
            selected_focus_areas = [lesson_focus_map.get(lid, 'unknown') for lid in selected_lessons]
            focus_area_count = sum(1 for fa in selected_focus_areas if fa in focus_areas)
            logger.info(f"LLM selected {len(selected_lessons)} lessons: {focus_area_count} from focus areas, {len(selected_lessons) - focus_area_count} from other areas")
            logger.info(f"Selected lesson focus areas: {selected_focus_areas}")
        if selected_activity:
            activity_focus = activity_focus_map.get(selected_activity, 'unknown')
            is_from_focus = activity_focus in focus_areas
            logger.info(f"LLM selected activity from {'focus area' if is_from_focus else 'other area'}: {activity_focus}")
        
        return (selected_lessons, selected_activity)
        
    except Exception as e:
        logger.error(f"Error in LLM content selection: {e}")
        raise


# ============================================================================
# Signup Guard Endpoint - Protect against spam account creation
# ============================================================================

@app.post("/signup-guard", response_model=SignupGuardResponse)
async def signup_guard(request: Request, payload: SignupGuardRequest):
    """
    Check whether a signup attempt should be allowed.

    This endpoint is called from the mobile app *before* creating a new account.
    It applies several safeguards:
      - Rate limits account creation per IP address (per hour / per day)
      - Rate limits account creation per device (if device_id is provided)
      - Blocks disposable / temporary email domains

    Returns:
      - allowed: True if signup can proceed
      - reason: Optional human-readable explanation if blocked
    """
    try:
        client_host = request.client.host if request.client else "unknown"

        allowed, reason = check_signup_guard(
            ip_address=client_host,
            email=payload.email,
            device_id=payload.device_id,
        )

        return SignupGuardResponse(
            allowed=allowed,
            reason=reason,
        )
    except Exception as e:
        logger.error(f"Error in signup_guard: {e}")
        # Fail safe: if guard fails, do NOT silently allow signups; be explicit
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to verify signup at the moment. Please try again later.",
        )


# Startup and Shutdown Events

@app.on_event("startup")
async def startup_event():
    """
    Runs when the FastAPI server starts.
    
    This is a good place to:
    - Initialize database connections
    - Load models
    - Check dependencies
    """
    global crisis_controller, emotion_predictor, llm_generator
    
    logger.info("Starting SehatMind Chatbot API server...")
    
    # Initialize crisis detection controller
    # SAFETY HIERARCHY: Crisis detection is initialized FIRST (highest priority)
    # SAFETY GUARDRAIL: Initialize the crisis detection system at startup
    # This controller is used to detect crisis situations before processing messages
    try:
        logger.info("Initializing crisis detection controller...")
        crisis_controller = DecisionController()
        logger.info("✓ Crisis detection controller initialized")
    except Exception as e:
        logger.error(f"⚠ Failed to initialize crisis detection controller: {e}")
        logger.warning("  Crisis detection will be disabled. This is a safety risk.")
        crisis_controller = None
    
    # Initialize emotion predictor
    # SAFETY HIERARCHY: Emotion model runs AFTER crisis detection (only if no crisis)
    try:
        logger.info("Initializing emotion detection model...")
        emotion_predictor = EmotionPredictor()
        logger.info("✓ Emotion detection model initialized")
    except Exception as e:
        logger.error(f"⚠ Failed to initialize emotion detection model: {e}")
        logger.warning("  Emotion detection will be disabled. Responses will not use emotion information.")
        emotion_predictor = None
    
    # Initialize LLM response generator
    # SAFETY HIERARCHY: LLM runs AFTER crisis detection (only if no crisis)
    try:
        logger.info("Initializing LLM response generator...")
        # Check if API key is available
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            logger.info(f"  GROQ_API_KEY found (length: {len(groq_key)})")
        else:
            logger.warning("  GROQ_API_KEY not found in environment variables!")
            logger.warning("  LLM will not work until API key is set and server is restarted.")
        llm_generator = LLMResponseGenerator()
        if llm_generator.api_key:
            logger.info("✓ LLM response generator initialized with API key")
        else:
            logger.warning("⚠ LLM response generator initialized but API key is missing!")
        logger.info("✓ LLM response generator initialized")
    except Exception as e:
        logger.error(f"⚠ Failed to initialize LLM response generator: {e}")
        logger.warning("  LLM response generation will be disabled. Fallback responses will be used.")
        llm_generator = None


@app.on_event("shutdown")
async def shutdown_event():
    """
    Runs when the FastAPI server shuts down.
    
    This is a good place to:
    - Close database connections
    - Clean up resources
    """
    logger.info("Shutting down SehatMind Chatbot API server...")


# Password Reset OTP Endpoints

@app.post("/auth/send-otp", response_model=SendOTPResponse)
async def send_otp(request: SendOTPRequest):
    """
    Send OTP to user's email for password reset.
    
    Args:
        request: SendOTPRequest with user's email
        
    Returns:
        SendOTPResponse indicating success or failure
    """
    try:
        email = request.email.strip().lower()
        
        # Validate email format
        if not OTPService.validate_email(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        # Generate OTP
        otp = OTPService.create_otp(email)
        
        # Send OTP via email
        email_service = EmailService()
        email_sent = email_service.send_otp_email(email, otp)
        
        if not email_sent:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send OTP email. Please try again later."
            )
        
        logger.info(f"OTP sent successfully to {email}")
        return SendOTPResponse(
            success=True,
            message="OTP sent to your email. Please check your inbox."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending OTP: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while sending OTP. Please try again."
        )


@app.post("/auth/verify-otp", response_model=VerifyOTPResponse)
async def verify_otp(request: VerifyOTPRequest):
    """
    Verify OTP code for password reset.
    
    Args:
        request: VerifyOTPRequest with email and OTP
        
    Returns:
        VerifyOTPResponse indicating if OTP is valid
    """
    try:
        email = request.email.strip().lower()
        otp = request.otp.strip()
        
        # Validate email format
        if not OTPService.validate_email(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        # Validate OTP format
        if not otp.isdigit() or len(otp) != 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OTP must be a 6-digit number"
            )
        
        # Verify OTP
        is_valid = OTPService.verify_otp(email, otp)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired OTP. Please request a new one."
            )
        
        logger.info(f"OTP verified successfully for {email}")
        return VerifyOTPResponse(
            success=True,
            message="OTP verified successfully. You can now reset your password."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying OTP: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while verifying OTP. Please try again."
        )


@app.post("/auth/reset-password", response_model=ResetPasswordResponse)
async def reset_password(request: ResetPasswordRequest):
    """
    Reset user's password after OTP verification.
    
    Note: This endpoint validates that OTP was verified.
    The actual password reset should be done through Firebase Auth.
    
    Args:
        request: ResetPasswordRequest with email and new password
        
    Returns:
        ResetPasswordResponse indicating success or failure
    """
    try:
        email = request.email.strip().lower()
        new_password = request.new_password
        
        # Validate email format
        if not OTPService.validate_email(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        # Validate password
        if len(new_password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )
        
        # Check if OTP was verified
        if not OTPService.is_otp_verified(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OTP not verified. Please verify your OTP first."
            )
        
        # Consume OTP (mark as used)
        OTPService.consume_otp(email)
        
        logger.info(f"Password reset request processed for {email}")
        return ResetPasswordResponse(
            success=True,
            message="Password reset verified. Please use Firebase Auth to update your password."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting password: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while resetting password. Please try again."
        )


# Error Handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom exception handler for HTTP exceptions.
    
    This ensures consistent error response format for the Flutter app.
    """
    return {
        "error": {
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    }


# Main execution
if __name__ == "__main__":
    """
    Run the server directly (for development).
    
    In production, use a proper ASGI server like uvicorn:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
    """
    import uvicorn
    
    # Get configuration from environment or use defaults
    # Railway uses PORT environment variable, fallback to API_PORT or 8000
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info("=" * 60)
    logger.info("IMPORTANT: Make sure your Flutter app uses the correct URL:")
    logger.info(f"  - If on same computer: http://127.0.0.1:{port}")
    logger.info(f"  - If on network: http://{host}:{port}")
    logger.info("=" * 60)
    
    try:
        uvicorn.run(
            "api.server:app",
            host=host,
            port=port,
            reload=True,  # Auto-reload on code changes (development only)
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")

