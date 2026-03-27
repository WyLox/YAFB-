"""Core processing module."""

from .device import detect_device
from .processor import process_img

__all__ = ["detect_device", "process_img"]
