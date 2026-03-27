"""File validation and security module."""

import os
from pathlib import Path
from fonctions.utils import logger, ALLOWED_MODES, MAX_FILE_SIZE


def validate_file_path(file_path: str) -> None:
    """Validate file path to prevent directory traversal attacks.

    Args:
        file_path: Path to validate

    Raises:
        ValueError: If path is invalid or traversal detected
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    try:
        abs_path = Path(file_path).resolve()
        allowed_dir = Path.cwd().resolve()

        try:
            abs_path.relative_to(allowed_dir)
        except ValueError:
            raise ValueError(
                f"Path traversal detected. Files must be in current directory or subdirectories. "
                f"Attempted: {abs_path}, Allowed: {allowed_dir}"
            )

        if ".." in file_path:
            raise ValueError("Path traversal (..) detected in file path")

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Invalid file path: {e}")


def check_file_exists_and_readable(file_path: str) -> None:
    """Check if file exists and is readable.

    Args:
        file_path: Path to check

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file isn't readable
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"File is not readable: {file_path}")


def check_file_size(file_path: str, max_size: int = MAX_FILE_SIZE) -> None:
    """Check if file size is within limits.

    Args:
        file_path: Path to check
        max_size: Maximum allowed size in bytes

    Raises:
        ValueError: If file is too large
    """
    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        raise ValueError(
            f"File too large: {size_mb:.1f}MB (limit: {max_mb:.1f}MB)"
        )


def validate_mode(mode: str) -> None:
    """Validate mode argument.

    Args:
        mode: Mode to validate

    Raises:
        ValueError: If mode is invalid
    """
    if mode not in ALLOWED_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Allowed modes: {', '.join(sorted(ALLOWED_MODES))}"
        )


def verify_model_exists(model_path: str) -> None:
    """Verify YOLO model file exists and is readable.

    Args:
        model_path: Path to YOLO model file

    Raises:
        FileNotFoundError: If model file not found
        PermissionError: If model file not readable
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"YOLO model file not found: {model_path}. "
            f"Please download a YOLOv8 model (e.g., yolov8n.pt or yolov8n-seg.pt)"
        )
    if not os.access(model_path, os.R_OK):
        raise PermissionError(f"Model file not readable: {model_path}")
    logger.info(f"YOLO model verified: {model_path}")
