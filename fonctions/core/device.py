"""Device detection module."""

import torch
from fonctions.utils import logger


def detect_device():
    """Detect best available device for inference.

    Returns:
        tuple: (device_type, device_name) - e.g., ('cuda', 'NVIDIA CUDA')
    """
    if torch.cuda.is_available():
        return 'cuda', 'NVIDIA CUDA'

    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        logger.debug(f"Available OpenVINO devices: {devices}")

        if any('GPU' in d for d in devices):
            for device in devices:
                if 'GPU.1' in device:
                    device_name = core.get_property(device, 'FULL_DEVICE_NAME')
                    logger.info(f"Discrete GPU found: {device_name}")
                    return 'openvino_gpu', f'Discrete GPU (OpenVINO) - {device_name}'

            gpu_device = [d for d in devices if 'GPU' in d][0]
            device_name = core.get_property(gpu_device, 'FULL_DEVICE_NAME')
            logger.info(f"GPU found: {device_name}")
            return 'openvino_gpu', f'GPU (OpenVINO) - {device_name}'
        elif 'CPU' in devices:
            return 'openvino_cpu', 'Intel CPU (OpenVINO optimized)'
    except ImportError:
        logger.debug("OpenVINO not available - install with: pip install openvino")
    except Exception as e:
        logger.debug(f"Error detecting OpenVINO devices: {e}")

    return 'cpu', 'CPU (no acceleration)'
