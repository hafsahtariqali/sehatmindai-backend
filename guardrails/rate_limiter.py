"""
Rate Limiter for Chat Messages

This module implements rate limiting to prevent exceeding Groq API free tier limits.
It tracks messages per user/session and enforces limits per time window.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Rate limit configuration (can be overridden via environment variables)
# Groq free tier typically allows ~30 requests per minute
MESSAGES_PER_MINUTE = int(os.getenv("RATE_LIMIT_MESSAGES_PER_MINUTE", "25"))  # Slightly below limit for safety
MESSAGES_PER_HOUR = int(os.getenv("RATE_LIMIT_MESSAGES_PER_HOUR", "500"))  # Daily limit buffer
MESSAGES_PER_DAY = int(os.getenv("RATE_LIMIT_MESSAGES_PER_DAY", "10000"))  # Very high daily limit

# Store request timestamps per session/user
# Key: session_id, Value: List of datetime timestamps
request_timestamps: Dict[str, List[datetime]] = defaultdict(list)


def cleanup_old_timestamps():
    """
    Remove timestamps older than 24 hours to prevent memory bloat.
    This should be called periodically.
    """
    now = datetime.now()
    cutoff = now - timedelta(hours=24)
    
    sessions_to_clean = []
    for session_id, timestamps in request_timestamps.items():
        # Keep only timestamps from last 24 hours
        filtered = [ts for ts in timestamps if ts > cutoff]
        if filtered:
            request_timestamps[session_id] = filtered
        else:
            sessions_to_clean.append(session_id)
    
    # Remove sessions with no recent activity
    for session_id in sessions_to_clean:
        del request_timestamps[session_id]
    
    if sessions_to_clean:
        logger.debug(f"Cleaned up {len(sessions_to_clean)} inactive sessions from rate limiter")


def check_rate_limit(session_id: str, user_wrote_urdu: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Check if the session has exceeded rate limits.
    
    Args:
        session_id: User session identifier
        user_wrote_urdu: Whether user is writing in Urdu (for localized error messages)
    
    Returns:
        Tuple of (is_allowed, error_message)
        - is_allowed: True if request is allowed, False if rate limited
        - error_message: None if allowed, error message string if rate limited
    """
    now = datetime.now()
    
    # Clean up old timestamps periodically (every 100 requests to avoid overhead)
    if len(request_timestamps) > 0 and len(request_timestamps) % 100 == 0:
        cleanup_old_timestamps()
    
    # Get timestamps for this session
    timestamps = request_timestamps[session_id]
    
    # Filter to relevant time windows
    one_minute_ago = now - timedelta(minutes=1)
    one_hour_ago = now - timedelta(hours=1)
    one_day_ago = now - timedelta(days=1)
    
    recent_minute = [ts for ts in timestamps if ts > one_minute_ago]
    recent_hour = [ts for ts in timestamps if ts > one_hour_ago]
    recent_day = [ts for ts in timestamps if ts > one_day_ago]
    
    # Check per-minute limit
    if len(recent_minute) >= MESSAGES_PER_MINUTE:
        wait_seconds = max(1, int(60 - (now - recent_minute[0]).total_seconds()))
        if user_wrote_urdu:
            return False, f"معاف کیجیے، آپ نے بہت زیادہ پیغامات بھیج دیے ہیں۔ براہ کرم {wait_seconds} سیکنڈ کا انتظار کریں، پھر دوبارہ کوشش کریں۔"
        else:
            return False, f"I understand you want to keep talking, but I need a moment to process. Please wait {wait_seconds} seconds before sending another message."
    
    # Check per-hour limit
    if len(recent_hour) >= MESSAGES_PER_HOUR:
        wait_minutes = max(1, int((60 - (now - recent_hour[0]).total_seconds() / 60)))
        if user_wrote_urdu:
            return False, f"معاف کیجیے، آپ نے اس گھنٹے میں بہت زیادہ پیغامات بھیج دیے ہیں۔ براہ کرم {wait_minutes} منٹ کا انتظار کریں، پھر دوبارہ کوشش کریں۔"
        else:
            return False, f"I'm here for you, but I need a short break. Please wait {wait_minutes} minutes before sending another message."
    
    # Check per-day limit
    if len(recent_day) >= MESSAGES_PER_DAY:
        if user_wrote_urdu:
            return False, "آج آپ نے بہت زیادہ پیغامات بھیج دیے ہیں۔ براہ کرم کل دوبارہ کوشش کریں۔ میں یہاں ہوں جب آپ واپس آئیں گے۔"
        else:
            return False, "You've sent many messages today. Please take a break and try again tomorrow. I'll be here when you return."
    
    # Request is allowed - record timestamp
    timestamps.append(now)
    
    # Keep only last 24 hours of timestamps for this session
    request_timestamps[session_id] = [ts for ts in timestamps if ts > one_day_ago]
    
    return True, None


def get_rate_limit_status(session_id: str) -> Dict[str, int]:
    """
    Get current rate limit status for a session.
    
    Args:
        session_id: User session identifier
    
    Returns:
        Dictionary with current usage counts:
        - messages_last_minute: Count in last minute
        - messages_last_hour: Count in last hour
        - messages_last_day: Count in last day
        - limit_per_minute: Per-minute limit
        - limit_per_hour: Per-hour limit
        - limit_per_day: Per-day limit
    """
    now = datetime.now()
    timestamps = request_timestamps.get(session_id, [])
    
    one_minute_ago = now - timedelta(minutes=1)
    one_hour_ago = now - timedelta(hours=1)
    one_day_ago = now - timedelta(days=1)
    
    return {
        "messages_last_minute": len([ts for ts in timestamps if ts > one_minute_ago]),
        "messages_last_hour": len([ts for ts in timestamps if ts > one_hour_ago]),
        "messages_last_day": len([ts for ts in timestamps if ts > one_day_ago]),
        "limit_per_minute": MESSAGES_PER_MINUTE,
        "limit_per_hour": MESSAGES_PER_HOUR,
        "limit_per_day": MESSAGES_PER_DAY,
    }

