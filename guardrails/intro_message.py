"""
Intro Message for New Chat Sessions

This module provides intro messages that are shown when a user starts a new chat session.
The message is personalized based on the user's preferred language and name.
"""

from typing import Optional


def get_intro_message_english(user_name: Optional[str] = None) -> str:
    """
    Returns an intro message in English for new chat sessions.
    
    Args:
        user_name: Optional user's first name. If provided, message is personalized.
    
    Returns:
        A personalized intro message in English
    """
    if user_name:
        return f"Hi {user_name}, how's your mood today?"
    else:
        return "Hi, how's your mood today?"


def get_intro_message_urdu(user_name: Optional[str] = None) -> str:
    """
    Returns an intro message in Urdu for new chat sessions.
    
    Args:
        user_name: Optional user's first name. If provided, message is personalized.
    
    Returns:
        A personalized intro message in Urdu
    """
    if user_name:
        return f"ہیلو {user_name}، آج کا دن کیسا گزر رہا ہے؟"
    else:
        return "ہیلو، آج کا دن کیسا گزر رہا ہے؟"

