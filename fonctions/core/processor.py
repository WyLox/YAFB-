"""Image processing module."""

import cv2
import numpy as np
from ultralytics import YOLO
from fonctions.utils import logger, BLUR_RADIUS, MIN_DETECTION_CONFIDENCE
from fonctions.core.device import detect_device

DEVICE, _ = detect_device()


def resize_image_to_fit(custom_img: any, target_w: int, target_h: int) -> any:
    """Resize custom image to fit target dimensions while maintaining aspect ratio.

    Args:
        custom_img: Custom image to resize
        target_w: Target width
        target_h: Target height

    Returns:
        Resized image fitting the target dimensions
    """
    if custom_img is None:
        return None

    h, w = custom_img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(custom_img, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def process_img(
    img: any,
    model: YOLO,
    blur_enabled: bool = True,
    custom_image_enabled: bool = False,
    custom_image: any = None
) -> any:
    """Process image to detect and blur faces using YOLOv8.

    Args:
        img: Input image (BGR format from OpenCV)
        model: YOLOv8 model instance
        blur_enabled: Whether to apply blur (default: True)
        custom_image_enabled: Whether to replace faces with custom image (default: False)
        custom_image: Custom image to use for replacement (default: None)

    Returns:
        Processed image with blurred faces or replaced with custom image

    Raises:
        ValueError: If image is invalid or processing fails
    """
    if img is None:
        raise ValueError("Input image is None")

    if img.size == 0:
        raise ValueError("Input image is empty")

    try:
        H, W = img.shape[:2]

        if H <= 0 or W <= 0:
            raise ValueError(f"Invalid image dimensions: {W}x{H}")

        if DEVICE.startswith('openvino'):
            results = model(img, conf=MIN_DETECTION_CONFIDENCE, verbose=False)
        else:
            results = model(img, conf=MIN_DETECTION_CONFIDENCE, verbose=False, device=DEVICE)

        faces_detected = False
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            if len(boxes) > 0:
                faces_detected = True
                logger.debug(f"Detected {len(boxes)} face(s)")

                if custom_image_enabled and custom_image is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        expand_factor = 0.25
                        w = x2 - x1
                        h = y2 - y1
                        w_expansion = int(w * expand_factor / 2)
                        h_expansion = int(h * expand_factor / 2)

                        x1 = x1 - w_expansion
                        y1 = y1 - h_expansion
                        x2 = x2 + w_expansion
                        y2 = y2 + h_expansion

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(W, x2)
                        y2 = min(H, y2)

                        if x2 > x1 and y2 > y1:
                            face_w = x2 - x1
                            face_h = y2 - y1

                            replacement_img = resize_image_to_fit(custom_image, face_w, face_h)

                            if replacement_img is not None:
                                img[y1:y2, x1:x2, :] = replacement_img

                    logger.info(f"Replaced {len(boxes)} face(s) with custom image")

                elif blur_enabled:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        expand_factor = 0.25
                        w = x2 - x1
                        h = y2 - y1
                        w_expansion = int(w * expand_factor / 2)
                        h_expansion = int(h * expand_factor / 2)

                        x1 = x1 - w_expansion
                        y1 = y1 - h_expansion
                        x2 = x2 + w_expansion
                        y2 = y2 + h_expansion

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(W, x2)
                        y2 = min(H, y2)

                        if x2 > x1 and y2 > y1:
                            img[y1:y2, x1:x2, :] = cv2.blur(
                                img[y1:y2, x1:x2, :],
                                (BLUR_RADIUS, BLUR_RADIUS)
                            )

                    logger.info(f"Blurred {len(boxes)} face(s)")
                else:
                    logger.debug(f"Blur disabled - showing {len(boxes)} face(s) without blur")

        if not faces_detected:
            if blur_enabled or custom_image_enabled:
                logger.warning("No faces detected - displaying black screen for security")
                img[:] = 0
            else:
                logger.debug("No faces detected - blur disabled, showing original frame")

        return img

    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
