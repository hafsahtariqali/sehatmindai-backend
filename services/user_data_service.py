"""
User Data Service - Fetch user data from Firestore

This service provides functions to fetch user data from Firestore,
specifically full names from the private collection for system use
(profile, chat personalization) while maintaining anonymization in main database.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_user_full_name(user_id: str) -> Optional[str]:
    """
    Fetch user's full name from Firestore private collection.
    
    The full name is stored in users/{user_id}/private/name for system use
    (profile display, chat personalization) while the main database only
    contains name abbreviations for anonymization.
    
    Args:
        user_id: Firebase user ID (UID)
    
    Returns:
        Full name string if found, None otherwise
    """
    try:
        # Try to import Firebase Admin SDK
        try:
            import firebase_admin
            from firebase_admin import firestore
            
            # Initialize Firebase if not already initialized
            if not firebase_admin._apps:
                # Check for credentials JSON string (Railway-friendly)
                cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
                if cred_json:
                    import json
                    cred_dict = json.loads(cred_json)
                    cred = firebase_admin.credentials.Certificate(cred_dict)
                    firebase_admin.initialize_app(cred)
                else:
                    # Check for credentials file path (local development)
                    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
                    if cred_path and Path(cred_path).exists():
                        cred = firebase_admin.credentials.Certificate(cred_path)
                        firebase_admin.initialize_app(cred)
                    else:
                        # Use default credentials (if running in Firebase environment)
                        firebase_admin.initialize_app()
            
            db = firestore.client()
            
            # Fetch full name from private collection
            doc_ref = db.collection('users').document(user_id).collection('private').document('name')
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                full_name = data.get('fullName') if data else None
                if full_name:
                    logger.debug(f"Retrieved full name for user {user_id}")
                    return full_name
                else:
                    logger.debug(f"No fullName found in private collection for user {user_id}")
                    return None
            else:
                logger.debug(f"Private name document does not exist for user {user_id}")
                return None
                
        except ImportError:
            # Firebase Admin SDK not available
            logger.debug("Firebase Admin SDK not available - cannot fetch user name from Firestore")
            return None
        except Exception as firebase_error:
            # Firebase error
            logger.warning(f"Firebase error fetching user name: {firebase_error}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching user full name: {e}")
        return None


def get_user_preferred_language(user_id: str) -> Optional[str]:
    """
    Fetch user's preferred language from Firestore.
    
    Args:
        user_id: Firebase user ID (UID)
    
    Returns:
        Preferred language string ("English" or "Urdu") if found, None otherwise
    """
    try:
        # Try to import Firebase Admin SDK
        try:
            import firebase_admin
            from firebase_admin import firestore
            
            # Initialize Firebase if not already initialized
            if not firebase_admin._apps:
                # Check for credentials JSON string (Railway-friendly)
                cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
                if cred_json:
                    import json
                    cred_dict = json.loads(cred_json)
                    cred = firebase_admin.credentials.Certificate(cred_dict)
                    firebase_admin.initialize_app(cred)
                else:
                    # Check for credentials file path (local development)
                    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
                    if cred_path and Path(cred_path).exists():
                        cred = firebase_admin.credentials.Certificate(cred_path)
                        firebase_admin.initialize_app(cred)
                    else:
                        firebase_admin.initialize_app()
            
            db = firestore.client()
            
            # Fetch preferred language from onboarding preferences
            doc_ref = db.collection('users').document(user_id).collection('onboarding').document('preferences')
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                language = data.get('language') if data else None
                if language:
                    logger.debug(f"Retrieved preferred language '{language}' for user {user_id}")
                    return language
                else:
                    logger.debug(f"No language preference found for user {user_id}")
                    return None
            else:
                logger.debug(f"Preferences document does not exist for user {user_id}")
                return None
                
        except ImportError:
            logger.debug("Firebase Admin SDK not available - cannot fetch language preference")
            return None
        except Exception as firebase_error:
            logger.warning(f"Firebase error fetching language preference: {firebase_error}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching user preferred language: {e}")
        return None

