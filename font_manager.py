"""
Font downloader for Noto Sans Bengali.
Called at app startup to ensure Bengali fonts are available for PDF rendering.
"""

import os
import urllib.request
import logging

logger = logging.getLogger(__name__)

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")

# Google Fonts CDN direct links for Noto Sans Bengali
FONT_URLS = {
    "NotoSansBengali-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/notosansbengali/"
        "NotoSansBengali%5Bwdth%2Cwght%5D.ttf"
    ),
}

# Fallback: DejaVu Sans (always available, limited Bangla support)
DEJAVU_URLS = {
    "DejaVuSans.ttf": "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf": "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf",
}


def ensure_fonts():
    """Download fonts if not already present. Called at app startup."""
    os.makedirs(FONT_DIR, exist_ok=True)

    downloaded_any = False

    # Try Noto Sans Bengali first
    for filename, url in FONT_URLS.items():
        dest = os.path.join(FONT_DIR, filename)
        if not os.path.exists(dest):
            try:
                logger.info(f"Downloading font: {filename}")
                urllib.request.urlretrieve(url, dest)
                logger.info(f"Downloaded {filename} successfully")
                downloaded_any = True
            except Exception as e:
                logger.warning(f"Could not download {filename}: {e}")

    # Also try DejaVu as fallback
    for filename, url in DEJAVU_URLS.items():
        dest = os.path.join(FONT_DIR, filename)
        if not os.path.exists(dest):
            try:
                logger.info(f"Downloading fallback font: {filename}")
                urllib.request.urlretrieve(url, dest)
                logger.info(f"Downloaded {filename} successfully")
            except Exception as e:
                logger.warning(f"Could not download {filename}: {e}")

    if downloaded_any:
        logger.info("Font download complete.")

    return os.listdir(FONT_DIR)
