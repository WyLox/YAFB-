"""Webcam processing mode."""

import cv2
from ultralytics import YOLO
from fonctions.utils import logger, DEFAULT_CAMERA_INDEX, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT
from fonctions.core import process_img


def process_webcam_mode(model: YOLO) -> None:
    """Process webcam stream in real-time.

    Args:
        model: YOLOv8 model instance

    Raises:
        RuntimeError: If camera fails or processing fails
    """
    logger.info("Starting webcam mode")

    cap = None
    try:
        cap = cv2.VideoCapture(DEFAULT_CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(
                "Failed to open camera. Is camera available and not in use by another application?"
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened successfully - Resolution: {actual_width}x{actual_height}")

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT)

        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                break

            try:
                frame = process_img(frame, model)
                cv2.imshow('frame', frame)
                frame_count += 1

                if frame_count % 30 == 0:
                    logger.debug(f"Processed {frame_count} frames")

            except Exception as e:
                logger.error(f"Error processing webcam frame: {e}")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exit key pressed")
                break

        logger.info(f"Webcam processing complete: {frame_count} frames processed")

    except Exception as e:
        logger.error(f"Webcam processing failed: {e}")
        raise RuntimeError(f"Webcam processing failed: {e}")

    finally:
        if cap is not None:
            cap.release()
            logger.debug("Camera released")
        cv2.destroyAllWindows()
        logger.debug("Windows closed")
