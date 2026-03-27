"""Video processing mode."""

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


def process_video_mode(file_path: str, output_dir: str, model: YOLO) -> None:
    """Process video file.

    Args:
        file_path: Path to input video
        output_dir: Output directory
        model: YOLOv8 model instance

    Raises:
        RuntimeError: If processing fails
    """
    logger.info(f"Processing video: {file_path}")

    cap = None
    output_video = None

    try:
        validate_file_path(file_path)
        check_file_exists_and_readable(file_path)
        check_file_size(file_path)

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {file_path}")

        logger.info("Video opened successfully")

        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read first frame from video")

        logger.info(f"Video loaded: {frame.shape[1]}x{frame.shape[0]}")

        output_path = os.path.join(output_dir, 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        output_video = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (frame.shape[1], frame.shape[0])
        )

        if not output_video.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")

        logger.info(f"Output video initialized: {output_path}")

        frame_count = 0
        while ret and frame is not None:
            frame_count += 1

            try:
                frame = process_img(frame, model)
                output_video.write(frame)

                if frame_count % 30 == 0:
                    logger.debug(f"Processed {frame_count} frames")

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                raise

            ret, frame = cap.read()

        logger.info(f"Video processing complete: {frame_count} frames processed")
        logger.info(f"Output saved to: {output_path}")

    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise RuntimeError(f"Video processing failed: {e}")

    finally:
        if cap is not None:
            cap.release()
            logger.debug("Video capture released")
        if output_video is not None:
            output_video.release()
            logger.debug("Video writer released")
