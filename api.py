"""
FastAPI Server for Chat Controller

Simple API endpoint that uses chat_controller.handle_user_message
to process user messages and return responses.

Designed for Flutter app integration.
"""

print("Loading API server...")

from fastapi import FastAPI, HTTPException, status
print("✓ FastAPI imported")
from fastapi.middleware.cors import CORSMiddleware
print("✓ CORS middleware imported")

from pydantic import BaseModel, Field
print("✓ Pydantic imported")

from typing import Optional
import sys
from pathlib import Path
print("✓ Standard libraries imported")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
print("✓ Path configured")

# Lazy import - chat_controller will be imported when needed
# This avoids loading PyTorch models at import time, which causes issues
# with uvicorn's reload mechanism on Windows
handle_user_message = None

def _lazy_import_chat_controller():
    """Lazy import chat_controller to avoid PyTorch loading issues on startup."""
    global handle_user_message
    if handle_user_message is None:
        from chat_controller import handle_user_message as _handle_user_message
        handle_user_message = _handle_user_message
    return handle_user_message

# Initialize FastAPI app
print("Creating FastAPI app...")
app = FastAPI(
    title="SehatMind Chat API",
    description="API for mental health chatbot",
    version="1.0.0"
)
print("✓ FastAPI app created")

# Configure CORS for Flutter app
print("Configuring CORS...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("✓ CORS configured")


# Request/Response Models

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="The user's message", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the user session. If not provided, a new session is created."
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="The bot's response text")
    crisis: bool = Field(..., description="Whether a crisis was detected")
    emotion: Optional[str] = Field(None, description="Detected emotion (if any)")


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "running", "message": "SehatMind Chat API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes user messages and returns bot responses.
    
    This endpoint:
    1. Calls handle_user_message from chat_controller
    2. Returns formatted response with crisis status and emotion
    
    Args:
        request: ChatRequest containing message and optional session_id
        
    Returns:
        ChatResponse with response text, crisis status, and detected emotion
        
    Raises:
        HTTPException: If message is invalid
    """
    # Validate message
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    
    # Generate session ID if not provided
    if not request.session_id:
        import uuid
        session_id = f"user_{uuid.uuid4().hex[:12]}"
    else:
        session_id = request.session_id
    
    # Call handle_user_message from chat_controller (lazy import)
    handle_user_message_func = _lazy_import_chat_controller()
    result = handle_user_message_func(
        user_text=request.message,
        session_id=session_id
    )
    
    # Extract response data
    response_text = result.get("response_text", "")
    is_crisis = result.get("is_crisis", False)
    detected_emotion = result.get("emotion_detected")
    
    # Return formatted response
    return ChatResponse(
        response=response_text,
        crisis=is_crisis,
        emotion=detected_emotion
    )


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Initializing SehatMind Chat API...")
    print("=" * 60)
    
    try:
        import uvicorn
        print("✓ Uvicorn imported")
        
        import os
        print("✓ OS module imported")
        
        # Get configuration from environment or use defaults
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", "8000"))
        print(f"✓ Configuration loaded: {host}:{port}")
        
        print("\n" + "=" * 60)
        print(f"Starting SehatMind Chat API on {host}:{port}")
        print("Connect your Flutter app to: http://localhost:8000/chat")
        print("\nPre-loading models (this may take 1-3 minutes)...")
        print("=" * 60)
        
        # Pre-load models at startup for faster first response
        try:
            print("\nLoading chat controller and models...")
            handle_user_message_func = _lazy_import_chat_controller()
            
            # Trigger model initialization by calling with a dummy message
            # This forces all models (crisis, emotion, response) to load
            print("  Initializing crisis detection model...")
            print("  Initializing emotion detection model...")
            print("  Initializing response engine...")
            print("  (This may take 1-3 minutes on first run)")
            
            # Dummy call to trigger model loading
            # Use a simple message that won't trigger crisis
            _ = handle_user_message_func("test", "startup_init")
            
            print("✓ All models loaded successfully")
            print("✓ Server ready - first message will be fast!")
        except Exception as e:
            print(f"⚠ Warning: Model pre-loading failed: {e}")
            print("  Models will load on first request instead")
            import traceback
            traceback.print_exc()
        
        print("=" * 60 + "\n")
        
        # Run uvicorn directly with app object
        # Disable reload on Windows to avoid multiprocessing issues with PyTorch
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,  # Disabled: causes issues with PyTorch on Windows
            log_level="info"
        )
    except Exception as e:
        print(f"\n❌ ERROR starting server: {e}")
        import traceback
        traceback.print_exc()
        raise

