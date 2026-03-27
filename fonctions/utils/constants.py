"""Configuration constants for Face Blurrer."""

import logging
import sys

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    return logging.getLogger(__name__)


logger = setup_logging()

ALLOWED_MODES = {"image", "video", "webcam", "gui"}

MAX_FILE_SIZE = 512 * 1024 * 1024

YOLO_MODEL_PATH = "yolov8n-face.pt"
BLUR_RADIUS = 70
MIN_DETECTION_CONFIDENCE = 0.5

DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_FPS = 30

DEFAULT_CUSTOM_IMAGE_PATH = None
CUSTOM_IMAGE_ENABLED = False
