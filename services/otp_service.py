"""
OTP Service for managing password reset OTPs
"""

import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
import re

logger = logging.getLogger(__name__)

# OTP configuration
OTP_LENGTH = 6
OTP_EXPIRY_MINUTES = 10
MAX_ATTEMPTS = 3

# In-memory storage for OTPs
# In production, consider using Redis or a database
_otp_storage: Dict[str, Dict] = {}


class OTPService:
    """Service for generating and validating OTPs"""
    
    @staticmethod
    def generate_otp() -> str:
        """Generate a random 6-digit OTP"""
        return ''.join([str(secrets.randbelow(10)) for _ in range(OTP_LENGTH)])
    
    @staticmethod
    def create_otp(email: str) -> str:
        """
        Create and store an OTP for the given email
        
        Args:
            email: User's email address
            
        Returns:
            The generated OTP code
        """
        # Clean up expired OTPs first
        OTPService._cleanup_expired()
        
        # Generate new OTP
        otp = OTPService.generate_otp()
        expires_at = datetime.now() + timedelta(minutes=OTP_EXPIRY_MINUTES)
        
        # Store OTP with metadata
        _otp_storage[email.lower()] = {
            'otp': otp,
            'expires_at': expires_at,
            'attempts': 0,
            'created_at': datetime.now()
        }
        
        logger.info(f"OTP created for {email}, expires at {expires_at}")
        return otp
    
    @staticmethod
    def verify_otp(email: str, otp: str) -> bool:
        """
        Verify an OTP for the given email
        
        Args:
            email: User's email address
            otp: OTP code to verify
            
        Returns:
            True if OTP is valid, False otherwise
        """
        email_lower = email.lower()
        
        # Check if OTP exists
        if email_lower not in _otp_storage:
            logger.warning(f"OTP verification failed: No OTP found for {email}")
            return False
        
        otp_data = _otp_storage[email_lower]
        
        # Check if OTP has expired
        if datetime.now() > otp_data['expires_at']:
            logger.warning(f"OTP verification failed: OTP expired for {email}")
            del _otp_storage[email_lower]
            return False
        
        # Check if max attempts exceeded
        if otp_data['attempts'] >= MAX_ATTEMPTS:
            logger.warning(f"OTP verification failed: Max attempts exceeded for {email}")
            del _otp_storage[email_lower]
            return False
        
        # Increment attempts
        otp_data['attempts'] += 1
        
        # Verify OTP
        if otp_data['otp'] == otp:
            # OTP is valid, but don't delete it yet (needed for password reset)
            logger.info(f"OTP verified successfully for {email}")
            otp_data['verified'] = True
            return True
        else:
            logger.warning(f"OTP verification failed: Invalid OTP for {email}")
            return False
    
    @staticmethod
    def is_otp_verified(email: str) -> bool:
        """
        Check if OTP has been verified for the given email
        
        Args:
            email: User's email address
            
        Returns:
            True if OTP exists and has been verified, False otherwise
        """
        email_lower = email.lower()
        
        if email_lower not in _otp_storage:
            return False
        
        otp_data = _otp_storage[email_lower]
        
        # Check if expired
        if datetime.now() > otp_data['expires_at']:
            del _otp_storage[email_lower]
            return False
        
        return otp_data.get('verified', False)
    
    @staticmethod
    def consume_otp(email: str) -> bool:
        """
        Consume (delete) an OTP after successful password reset
        
        Args:
            email: User's email address
            
        Returns:
            True if OTP was consumed, False if not found or not verified
        """
        email_lower = email.lower()
        
        if email_lower not in _otp_storage:
            return False
        
        otp_data = _otp_storage[email_lower]
        
        # Only consume if verified
        if not otp_data.get('verified', False):
            return False
        
        # Delete OTP
        del _otp_storage[email_lower]
        logger.info(f"OTP consumed for {email}")
        return True
    
    @staticmethod
    def _cleanup_expired():
        """Remove expired OTPs from storage"""
        now = datetime.now()
        expired_emails = [
            email for email, data in _otp_storage.items()
            if now > data['expires_at']
        ]
        for email in expired_emails:
            del _otp_storage[email]
        
        if expired_emails:
            logger.info(f"Cleaned up {len(expired_emails)} expired OTPs")
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

