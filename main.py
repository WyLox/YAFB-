"""Face Blurrer - Real-time face detection and blurring assistant.

This application detects and blurs faces in images, videos, and webcam feeds.
Modes: image, video, webcam, gui
Output: ./output/ (output.png or output.mp4)
Security: Black screen if no faces detected (prevents face leakage)
"""

import os
import sys
import argparse
import torch
from ultralytics import YOLO
from fonctions.utils import (
    logger,
    ALLOWED_MODES,
    YOLO_MODEL_PATH,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_CAMERA_HEIGHT,
)
from fonctions.validation import validate_mode, verify_model_exists
from fonctions.core import detect_device
from fonctions.modes import (
    process_image_mode,
    process_video_mode,
    process_webcam_mode,
    process_gui_mode,
)

DEVICE, DEVICE_NAME = detect_device()


def load_yolo_model(model_path: str) -> YOLO:
    """Load YOLO model with appropriate device configuration.

    Args:
        model_path: Path to YOLO model file

    Returns:
        Loaded YOLO model

    Raises:
        RuntimeError: If model fails to load
    """
    try:
        logger.info(f"Loading YOLOv8 model on {DEVICE_NAME}...")

        if DEVICE.startswith('openvino'):
            if DEVICE == 'openvino_gpu':
                logger.info("Configuring OpenVINO to use GPU...")
                os.environ['OPENVINO_DEFAULT_DEVICE'] = 'GPU.1'
            else:
                os.environ['OPENVINO_DEFAULT_DEVICE'] = 'CPU'

            openvino_model_path = model_path.replace('.pt', '_openvino_model')

            if not os.path.exists(openvino_model_path):
                logger.info("Exporting model to OpenVINO format (one-time operation)...")
                temp_model = YOLO(model_path)
                temp_model.export(format='openvino', half=False)
                logger.info(f"Model exported to: {openvino_model_path}")

            model = YOLO(openvino_model_path, task='detect')

            if DEVICE == 'openvino_gpu':
                model.openvino_device = 'GPU.1'
                logger.info("GPU.1 (Discrete GPU) device selected for OpenVINO inference")
            else:
                model.openvino_device = 'CPU'
                logger.info("CPU device selected for OpenVINO inference")

            try:
                import openvino as ov
                core = ov.Core()
                devices = core.available_devices
                logger.info(f"Available OpenVINO devices: {devices}")
                for d in devices:
                    info = core.get_property(d, 'FULL_DEVICE_NAME')
                    logger.info(f"  - {d}: {info}")
            except Exception as e:
                logger.debug(f"Could not enumerate devices: {e}")

        elif DEVICE == 'cuda':
            model = YOLO(model_path)
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            logger.info(f"NVIDIA GPU detected: {gpu_name}")

        else:
            model = YOLO(model_path)
            logger.warning("No GPU acceleration - running on CPU (slower)")

        logger.info("YOLOv8 model loaded successfully")
        logger.info(f"Using device: {DEVICE_NAME}")
        return model

    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Model loading failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise RuntimeError(f"Failed to load YOLO model: {e}")


def prepare_output_directory(output_dir: str) -> None:
    """Create and verify output directory.

    Args:
        output_dir: Path to output directory

    Raises:
        RuntimeError: If directory creation or permission check fails
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"No write permission for output directory: {output_dir}")
        logger.info(f"Output directory ready: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to prepare output directory: {e}")
        raise RuntimeError(f"Failed to prepare output directory: {e}")


def main():
    """Main entry point."""
    try:
        parser = argparse.ArgumentParser(
            description="Face blurring tool using YOLOv8"
        )
        parser.add_argument(
            "--mode",
            default='gui',
            help="Processing mode: image, video, webcam, or gui (default: gui)"
        )
        parser.add_argument(
            "--filePath",
            default=None,
            help="Path to input file (required for image and video modes)"
        )
        args = parser.parse_args()

        logger.info(f"Starting face blurrer (mode: {args.mode})")

        validate_mode(args.mode)

        output_dir = './output'
        prepare_output_directory(output_dir)

        verify_model_exists(YOLO_MODEL_PATH)
        model = load_yolo_model(YOLO_MODEL_PATH)

        try:
            if args.mode == "image":
                if not args.filePath:
                    raise ValueError("--filePath is required for image mode")
                process_image_mode(args.filePath, output_dir, model)

            elif args.mode == "video":
                if not args.filePath:
                    raise ValueError("--filePath is required for video mode")
                process_video_mode(args.filePath, output_dir, model)

            elif args.mode == "webcam":
                process_webcam_mode(model)

            elif args.mode == "gui":
                process_gui_mode(model)

            logger.info("Processing completed successfully")

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise RuntimeError(f"Processing failed: {e}")

        finally:
            logger.debug("YOLO model resources released")

    except (ValueError, RuntimeError) as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
