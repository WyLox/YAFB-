"""Image processing mode."""

import os
import cv2
from ultralytics import YOLO
from fonctions.utils import logger
from fonctions.validation import (
    validate_file_path,
    check_file_exists_and_readable,
    check_file_size,
)
from fonctions.core import process_img


def process_image_mode(file_path: str, output_dir: str, model: YOLO) -> None:
    """Process single image file.

    Args:
        file_path: Path to input image
        output_dir: Output directory
        model: YOLOv8 model instance

    Raises:
        RuntimeError: If processing fails
    """
    logger.info(f"Processing image: {file_path}")

    try:
        validate_file_path(file_path)
        check_file_exists_and_readable(file_path)
        check_file_size(file_path)

        img = cv2.imread(file_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {file_path}")

        logger.info(f"Image loaded: {img.shape[1]}x{img.shape[0]}")

        img = process_img(img, model)

        output_path = os.path.join(output_dir, 'output.png')
        if not cv2.imwrite(output_path, img):
            raise RuntimeError(f"Failed to write output image: {output_path}")

        logger.info(f"Output saved to: {output_path}")

    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise RuntimeError(f"Image processing failed: {e}")
