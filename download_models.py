"""
Script to download model files for Railway deployment.
Downloads models from Google Drive during Railway build.
"""
import os
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs from environment variables (set in Railway)
CRISIS_MODEL_URL = os.getenv("CRISIS_MODEL_URL", "")
EMOTION_MODEL_URL = os.getenv("EMOTION_MODEL_URL", "")

CRISIS_MODEL_PATH = Path("crisis_model/model/model.safetensors")
EMOTION_MODEL_PATH = Path("emotion_model/model/model.safetensors")


def download_from_google_drive(url: str, filepath: Path):
    """
    Download a file from Google Drive.
    Handles both direct download links and shareable links.
    """
    if not url:
        logger.warning(f"No URL provided for {filepath}")
        return False
    
    logger.info(f"Downloading {filepath.name} from Google Drive...")
    
    try:
        # Convert Google Drive shareable link to direct download link
        # Format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
        # Convert to: https://drive.google.com/uc?export=download&id=FILE_ID
        if "drive.google.com" in url:
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            else:
                download_url = url
        else:
            download_url = url
        
        # Download with session to handle large files and virus scan warnings
        session = requests.Session()
        response = session.get(download_url, stream=True, timeout=600, allow_redirects=True)
        
        # Check if Google Drive shows virus scan warning
        if "virus scan warning" in response.text.lower() or "download_warning" in response.url:
            # Extract confirm token and retry
            confirm_token = None
            for line in response.text.split('\n'):
                if "download_warning" in line:
                    confirm_token = line.split("download_warning(")[1].split(",")[0].strip("'\"")
                    break
            
            if confirm_token:
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                response = session.get(download_url, stream=True, timeout=600)
        
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file in chunks
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                            logger.info(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Successfully downloaded {filepath.name} ({file_size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download {filepath.name}: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Downloading model files for Railway deployment...")
    logger.info("=" * 60)
    
    success = True
    
    # Download crisis model
    if not CRISIS_MODEL_PATH.exists():
        if CRISIS_MODEL_URL:
            if not download_from_google_drive(CRISIS_MODEL_URL, CRISIS_MODEL_PATH):
                success = False
        else:
            logger.warning("CRISIS_MODEL_URL not set. Skipping crisis model download.")
    else:
        logger.info(f"✓ Crisis model already exists: {CRISIS_MODEL_PATH}")
    
    # Download emotion model
    if not EMOTION_MODEL_PATH.exists():
        if EMOTION_MODEL_URL:
            if not download_from_google_drive(EMOTION_MODEL_URL, EMOTION_MODEL_PATH):
                success = False
        else:
            logger.warning("EMOTION_MODEL_URL not set. Skipping emotion model download.")
    else:
        logger.info(f"✓ Emotion model already exists: {EMOTION_MODEL_PATH}")
    
    if success:
        logger.info("=" * 60)
        logger.info("✓ All model files downloaded successfully!")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("✗ Some model files failed to download!")
        logger.error("=" * 60)
        exit(1)

