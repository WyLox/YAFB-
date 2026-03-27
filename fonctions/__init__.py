"""Face Blurrer - Modular functions package."""

from .core.device import detect_device
from .core.processor import process_img
from .modes.image_mode import process_image_mode
from .modes.video_mode import process_video_mode
from .modes.webcam_mode import process_webcam_mode
from .modes.gui_mode import process_gui_mode, FaceBlurrerGUI
from .validation.file_validator import (
    validate_file_path,
    check_file_exists_and_readable,
    check_file_size,
    validate_mode,
    verify_model_exists,
)

__all__ = [
    "detect_device",
    "process_img",
    "process_image_mode",
    "process_video_mode",
    "process_webcam_mode",
    "process_gui_mode",
    "FaceBlurrerGUI",
    "validate_file_path",
    "check_file_exists_and_readable",
    "check_file_size",
    "validate_mode",
    "verify_model_exists",
]
