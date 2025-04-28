import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Database
DATABASE_PATH = BASE_DIR / "image_db.sqlite"

# File uploads
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Model paths
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
CLIP_MODEL = "ViT-B/32"

# App config
SECRET_KEY = "fsudyfgasjfbdsahfga12322312"

# Add default categories for image classification
DEFAULT_CATEGORIES = [
    "person",
    "animal",
    "vehicle",
    "building",
    "nature",
    "food",
    "technology",
    "sports",
    "furniture",
    "clothing"
]

