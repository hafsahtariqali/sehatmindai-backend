"""
Script to download model files for Railway deployment.
Run this during Railway build if models are not in the repository.
"""
import os
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs - you'll need to upload models to cloud storage (Google Drive, Dropbox, etc.)
# For now, this is a placeholder. You'll need to:
# 1. Upload model.safetensors files to cloud storage
# 2. Get direct download links
# 3. Update the URLs below

CRISIS_MODEL_URL = os.getenv("CRISIS_MODEL_URL", "")
EMOTION_MODEL_URL = os.getenv("EMOTION_MODEL_URL", "")

CRISIS_MODEL_PATH = Path("crisis_model/model/model.safetensors")
EMOTION_MODEL_PATH = Path("emotion_model/model/model.safetensors")


def download_file(url: str, filepath: Path):
    """Download a file from URL to filepath."""
    if not url:
        logger.warning(f"No URL provided for {filepath}")
        return False
    
    logger.info(f"Downloading {filepath} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filepath}: {e}")
        return False


if __name__ == "__main__":
    # Only download if files don't exist
    if not CRISIS_MODEL_PATH.exists() and CRISIS_MODEL_URL:
        download_file(CRISIS_MODEL_URL, CRISIS_MODEL_PATH)
    
    if not EMOTION_MODEL_PATH.exists() and EMOTION_MODEL_URL:
        download_file(EMOTION_MODEL_URL, EMOTION_MODEL_PATH)

