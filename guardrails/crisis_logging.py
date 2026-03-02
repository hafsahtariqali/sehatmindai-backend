"""
Crisis Detection Logging Service

ANONYMIZED RESEARCH LOGGING:
This module logs crisis detection events for research analysis purposes.
The logging is anonymized - NO raw user text is stored.
Only metadata is logged: timestamp, detection method, and detection labels.

Data is stored securely for research analysis to improve crisis detection systems.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)


class CrisisLoggingService:
    """
    Service for logging crisis detection events for anonymized research analysis.
    
    IMPORTANT: This service does NOT store raw user text.
    Only metadata (timestamp, detection labels) is logged for research purposes.
    """
    
    def __init__(self, log_file_path: Optional[Path] = None):
        """
        Initialize the crisis logging service.
        
        Args:
            log_file_path: Optional path to log file. If None, uses default location.
        """
        if log_file_path is None:
            # Default to logs directory in project root
            base_dir = Path(__file__).parent.parent.parent
            log_file_path = base_dir / "logs" / "crisis_detections.jsonl"
        
        self.log_file_path = Path(log_file_path)
        
        # Create logs directory if it doesn't exist
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _determine_detection_label(self, detection_result: Dict) -> str:
        """
        Determine the detection label from the crisis detection result.
        
        Args:
            detection_result: The result dictionary from crisis detection
            
        Returns:
            A string label describing how the crisis was detected
        """
        if detection_result.get("overridden_by_whitelist", False):
            return "crisis_detected_by_whitelist"
        elif detection_result.get("overridden_by_guardrails", False):
            severity = detection_result.get("guardrail_severity", "unknown")
            return f"crisis_detected_by_guardrails_{severity}"
        else:
            # Detected by ML model
            confidence = detection_result.get("confidence", 0.0)
            return f"crisis_detected_by_model_confidence_{confidence:.2f}"
    
    def log_crisis_detection(
        self,
        session_id: str,
        detection_result: Dict,
        detection_method: Optional[str] = None
    ) -> None:
        """
        Log a crisis detection event for anonymized research analysis.
        
        ANONYMIZED RESEARCH LOGGING:
        This function logs crisis detection events WITHOUT storing raw user text.
        Only metadata is logged: timestamp, session_id (anonymized), detection label.
        
        Data stored:
        - timestamp: When the crisis was detected
        - session_id: Anonymized session identifier (not personally identifiable)
        - detection_label: How the crisis was detected (whitelist, guardrails, model)
        - confidence: Detection confidence score (if available)
        - detection_method: Additional detection method information (if provided)
        
        Data NOT stored:
        - Raw user text (privacy-protected)
        - Personal identifiers
        - User messages
        
        Args:
            session_id: Anonymized session identifier
            detection_result: The result dictionary from crisis detection
            detection_method: Optional additional detection method information
        """
        try:
            # Determine detection label
            detection_label = self._determine_detection_label(detection_result)
            
            # Extract metadata (NO raw user text)
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "session_id": session_id,  # Anonymized identifier
                "detection_label": detection_label,
                "confidence": detection_result.get("confidence"),
                "overridden_by_whitelist": detection_result.get("overridden_by_whitelist", False),
                "overridden_by_guardrails": detection_result.get("overridden_by_guardrails", False),
            }
            
            # Add guardrail metadata if available
            if detection_result.get("guardrail_severity"):
                log_entry["guardrail_severity"] = detection_result.get("guardrail_severity")
                log_entry["risk_score"] = detection_result.get("risk_score")
            
            # Add detection method if provided
            if detection_method:
                log_entry["detection_method"] = detection_method
            
            # Write to JSONL file (one JSON object per line)
            # JSONL format is efficient for append-only logging
            # Use ensure_ascii=False to preserve Unicode characters (Urdu script, etc.)
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
            logger.info(f"Crisis detection logged: {detection_label} (session: {session_id})")
            
        except Exception as e:
            # Log errors but don't fail crisis detection
            logger.error(f"Error logging crisis detection: {e}")
    
    def log_to_firebase(
        self,
        session_id: str,
        detection_result: Dict,
        detection_method: Optional[str] = None
    ) -> None:
        """
        Log crisis detection to Firebase Firestore (if Firebase is configured).
        
        ANONYMIZED RESEARCH LOGGING:
        This function logs to Firebase WITHOUT storing raw user text.
        Only metadata is stored for research analysis.
        
        Args:
            session_id: Anonymized session identifier
            detection_result: The result dictionary from crisis detection
            detection_method: Optional additional detection method information
            
        Note:
            This method requires Firebase Admin SDK to be installed and configured.
            If Firebase is not available, logs to file instead.
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
                
                # Determine detection label
                detection_label = self._determine_detection_label(detection_result)
                
                # Create anonymized log entry (NO raw user text)
                log_data = {
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "session_id": session_id,  # Anonymized identifier
                    "detection_label": detection_label,
                    "confidence": detection_result.get("confidence"),
                    "overridden_by_whitelist": detection_result.get("overridden_by_whitelist", False),
                    "overridden_by_guardrails": detection_result.get("overridden_by_guardrails", False),
                }
                
                # Add guardrail metadata if available
                if detection_result.get("guardrail_severity"):
                    log_data["guardrail_severity"] = detection_result.get("guardrail_severity")
                    log_data["risk_score"] = detection_result.get("risk_score")
                
                # Add detection method if provided
                if detection_method:
                    log_data["detection_method"] = detection_method
                
                # Store in Firestore collection
                collection_name = "crisis_detections_research"  # Research collection
                db.collection(collection_name).add(log_data)
                
                logger.info(f"Crisis detection logged to Firebase: {detection_label} (session: {session_id})")
                
            except ImportError:
                # Firebase Admin SDK not available, fall back to file logging
                logger.debug("Firebase Admin SDK not available, using file logging")
                self.log_crisis_detection(session_id, detection_result, detection_method)
            except Exception as firebase_error:
                # Firebase error, fall back to file logging
                logger.warning(f"Firebase logging failed, using file logging: {firebase_error}")
                self.log_crisis_detection(session_id, detection_result, detection_method)
                
        except Exception as e:
            # Log errors but don't fail crisis detection
            logger.error(f"Error logging crisis detection to Firebase: {e}")


# Global logging service instance
_logging_service: Optional[CrisisLoggingService] = None


def get_crisis_logging_service() -> CrisisLoggingService:
    """
    Get or create the global crisis logging service instance.
    
    Returns:
        CrisisLoggingService instance
    """
    global _logging_service
    if _logging_service is None:
        _logging_service = CrisisLoggingService()
    return _logging_service


def log_crisis_detection(
    session_id: str,
    detection_result: Dict,
    use_firebase: bool = False,
    detection_method: Optional[str] = None
) -> None:
    """
    Convenience function to log crisis detection events.
    
    ANONYMIZED RESEARCH LOGGING:
    Logs crisis detection WITHOUT storing raw user text.
    Only metadata is logged for research analysis.
    
    Args:
        session_id: Anonymized session identifier
        detection_result: The result dictionary from crisis detection
        use_firebase: If True, attempt to log to Firebase (falls back to file if unavailable)
        detection_method: Optional additional detection method information
    """
    service = get_crisis_logging_service()
    
    if use_firebase:
        service.log_to_firebase(session_id, detection_result, detection_method)
    else:
        service.log_crisis_detection(session_id, detection_result, detection_method)

