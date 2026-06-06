"""
Firebase Auth helpers (Admin SDK) for password reset and user lookup.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PASSWORD_MIN_LENGTH = 8
_PASSWORD_HAS_NUMBER = re.compile(r"[0-9]")
_PASSWORD_HAS_SPECIAL = re.compile(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;/`~]')


def _ensure_firebase_initialized() -> None:
    import firebase_admin

    if firebase_admin._apps:
        return

    cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if cred_json:
        cred_dict = json.loads(cred_json)
        cred = firebase_admin.credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        return

    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    if cred_path and Path(cred_path).exists():
        cred = firebase_admin.credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        return

    firebase_admin.initialize_app()


def validate_password_strength(password: str) -> Optional[str]:
    """Return an error message if password is weak, else None."""
    if len(password) < _PASSWORD_MIN_LENGTH:
        return f"Password must be at least {_PASSWORD_MIN_LENGTH} characters long"
    if not _PASSWORD_HAS_NUMBER.search(password):
        return "Password must include at least one number"
    if not _PASSWORD_HAS_SPECIAL.search(password):
        return "Password must include at least one special character"
    return None


def user_exists_by_email(email: str) -> bool:
    """Return True if a Firebase Auth user exists for this email."""
    from firebase_admin import auth

    _ensure_firebase_initialized()
    try:
        auth.get_user_by_email(email.strip().lower())
        return True
    except auth.UserNotFoundError:
        return False


def update_password_by_email(email: str, new_password: str) -> None:
    """Set a new password for the Firebase Auth user with this email."""
    from firebase_admin import auth

    _ensure_firebase_initialized()
    normalized = email.strip().lower()

    try:
        user = auth.get_user_by_email(normalized)
    except auth.UserNotFoundError as e:
        raise ValueError("No account found for this email address.") from e

    auth.update_user(user.uid, password=new_password)
    logger.info(f"Password updated for user {user.uid}")
