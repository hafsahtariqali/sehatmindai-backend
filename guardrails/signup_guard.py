"""
Signup Guard - Detect and limit spam account creation

This module implements simple, in-memory heuristics to detect and block
potentially abusive signup patterns, such as:
 - Too many accounts from the same IP in a short time window
 - Too many accounts from the same device ID
 - Disposable / temporary email domains

For production, consider replacing the in-memory store with Redis/DB.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)

# In-memory tracking structures
_signup_attempts_by_ip: Dict[str, List[datetime]] = defaultdict(list)
_signup_attempts_by_device: Dict[str, List[datetime]] = defaultdict(list)
_blocked_ips: Dict[str, datetime] = {}  # IP -> block_until

# Configuration (can be tuned or moved to env vars)
MAX_SIGNUPS_PER_IP_HOUR = 5
MAX_SIGNUPS_PER_IP_DAY = 20
MAX_SIGNUPS_PER_DEVICE_DAY = 5
BLOCK_DURATION_HOURS = 6

# Simple disposable email domains list (extend as needed)
DISPOSABLE_DOMAINS = {
    "mailinator.com",
    "10minutemail.com",
    "guerrillamail.com",
    "tempmail.com",
    "yopmail.com",
    "discard.email",
}


def _is_disposable_email(email: str) -> bool:
    """Check if email domain is in a disposable list."""
    try:
        domain = email.split("@", 1)[1].lower()
    except IndexError:
        return True  # Invalid email format is treated as suspicious
    return domain in DISPOSABLE_DOMAINS


def _cleanup_old_attempts() -> None:
    """Remove attempts older than 24 hours to prevent memory bloat."""
    now = datetime.utcnow()
    cutoff = now - timedelta(days=1)

    for store in (_signup_attempts_by_ip, _signup_attempts_by_device):
        keys_to_delete = []
        for key, timestamps in store.items():
            filtered = [ts for ts in timestamps if ts > cutoff]
            if filtered:
                store[key] = filtered
            else:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del store[key]

    # Cleanup expired IP blocks
    expired_blocks = [ip for ip, until in _blocked_ips.items() if until <= now]
    for ip in expired_blocks:
        del _blocked_ips[ip]


def check_signup_guard(
    ip_address: str,
    email: str,
    device_id: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check whether a signup attempt should be allowed.

    Args:
        ip_address: Client IP address.
        email: Email used for signup.
        device_id: Optional client-side device identifier.

    Returns:
        (allowed, reason)
        - allowed: True if signup is allowed, False if blocked.
        - reason: Optional human-readable reason if blocked.
    """
    now = datetime.utcnow()

    # Periodic cleanup
    _cleanup_old_attempts()

    # Check if IP is currently blocked
    if ip_address in _blocked_ips:
        block_until = _blocked_ips[ip_address]
        if now < block_until:
            wait_hours = max(1, int((block_until - now).total_seconds() // 3600))
            return False, (
                f"Too many accounts were created from this network. "
                f"Please try again in about {wait_hours} hour(s)."
            )
        else:
            # Block expired
            del _blocked_ips[ip_address]

    # Basic email validation
    email = email.strip()
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        return False, "Please use a valid email address."

    # Check for disposable email domains
    if _is_disposable_email(email):
        return False, "Disposable email addresses are not allowed. Please use a real email address."

    # Track attempts by IP
    ip_attempts = _signup_attempts_by_ip[ip_address]
    ip_attempts.append(now)

    # Attempts in last hour/day
    one_hour_ago = now - timedelta(hours=1)
    one_day_ago = now - timedelta(days=1)

    attempts_last_hour = [ts for ts in ip_attempts if ts > one_hour_ago]
    attempts_last_day = [ts for ts in ip_attempts if ts > one_day_ago]

    if len(attempts_last_hour) > MAX_SIGNUPS_PER_IP_HOUR:
        # Soft block for a few hours
        _blocked_ips[ip_address] = now + timedelta(hours=BLOCK_DURATION_HOURS)
        return False, (
            "We've detected many accounts being created from this network in a short time. "
            "Please wait a few hours before trying again."
        )

    if len(attempts_last_day) > MAX_SIGNUPS_PER_IP_DAY:
        _blocked_ips[ip_address] = now + timedelta(hours=BLOCK_DURATION_HOURS)
        return False, (
            "We've detected unusually high signup activity from this network. "
            "Please try again later."
        )

    # Track attempts by device (if provided)
    if device_id:
        device_attempts = _signup_attempts_by_device[device_id]
        device_attempts.append(now)
        device_attempts_last_day = [ts for ts in device_attempts if ts > one_day_ago]

        if len(device_attempts_last_day) > MAX_SIGNUPS_PER_DEVICE_DAY:
            return False, (
                "Too many accounts were created from this device today. "
                "Please use your existing account or try again tomorrow."
            )

    # If we reach here, signup is allowed
    return True, None


