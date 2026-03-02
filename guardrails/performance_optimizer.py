"""
Performance Optimizer - Caching and optimization utilities

This module provides caching and optimization functions to improve response times.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

# Cache for user data (name, language) to avoid repeated Firestore calls
# Key: session_id, Value: dict with 'name', 'language', 'cached_at'
_user_data_cache: Dict[str, Dict] = {}
CACHE_TTL_SECONDS = 300  # Cache for 5 minutes


def get_cached_user_data(session_id: str) -> Optional[Dict]:
    """
    Get cached user data (name, language) for a session.
    
    Args:
        session_id: User session identifier
    
    Returns:
        Dictionary with 'name' and 'language' if cached and not expired, None otherwise
    """
    if session_id not in _user_data_cache:
        return None
    
    cached_data = _user_data_cache[session_id]
    cache_age = (datetime.now() - cached_data['cached_at']).total_seconds()
    
    if cache_age > CACHE_TTL_SECONDS:
        # Cache expired, remove it
        del _user_data_cache[session_id]
        return None
    
    return {
        'name': cached_data.get('name'),
        'language': cached_data.get('language')
    }


def cache_user_data(session_id: str, name: Optional[str] = None, language: Optional[str] = None):
    """
    Cache user data for faster access.
    
    Args:
        session_id: User session identifier
        name: User's name (optional)
        language: User's preferred language (optional)
    """
    if session_id not in _user_data_cache:
        _user_data_cache[session_id] = {}
    
    if name is not None:
        _user_data_cache[session_id]['name'] = name
    if language is not None:
        _user_data_cache[session_id]['language'] = language
    
    _user_data_cache[session_id]['cached_at'] = datetime.now()


def cleanup_expired_cache():
    """
    Remove expired cache entries to prevent memory bloat.
    """
    now = datetime.now()
    expired_sessions = []
    
    for session_id, data in _user_data_cache.items():
        cache_age = (now - data['cached_at']).total_seconds()
        if cache_age > CACHE_TTL_SECONDS:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del _user_data_cache[session_id]
    
    if expired_sessions:
        logger.debug(f"Cleaned up {len(expired_sessions)} expired cache entries")

