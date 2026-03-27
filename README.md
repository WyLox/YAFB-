# Face Blurrer

A simple, fast face detection and blurring tool powered by YOLOv8 and GPU acceleration.

## What It Does

Detects faces in images, videos, and webcam feeds, then blurs or replaces them. Perfect for privacy protection.

**Features:**
-  Works with images, videos, and live webcam
-  GPU-accelerated (NVIDIA, Intel GPU, or CPU fallback)
-  GUI, command-line, and programmatic modes
-  Black screen if no faces detected (prevents face leakage)

## Quick Start

### 1. Setup (One-time)

**Linux/Fedora:**
```bash
./setup.sh
```

**Windows:**
```bash
setup.bat
```

### 2. Run

```bash
python main.py
```

Opens the GUI by default.

## Command Line Usage

```bash
# GUI mode
python main.py

# Process an image
python main.py --mode image --filePath photo.jpg

# Process a video
python main.py --mode video --filePath video.mp4

# Use your webcam
python main.py --mode webcam
```

Results go to `./output/`

## What Gets Installed

- YOLOv8 (face detection model)
- PyTorch (deep learning)
- OpenCV (image processing)
- OpenVINO (GPU support)

## Requirements

- Python 3.8+
- GPU recommended (Intel iGPU, NVIDIA, or AMD via OpenVINO)
- ~2GB disk space for model

## GPU Support

Auto-detects and uses:
- NVIDIA GPUs (CUDA)
- Intel GPUs (iGPU/Arc via OpenVINO)
- Falls back to CPU if no GPU found

See logs on startup for detected device.

## Troubleshooting

**Python not found?**
- Install: https://www.python.org/downloads/

**GPU not detected?**
- The app logs detected devices on startup
- It works fine on CPU, just slower

**Import errors?**
- Ensure virtual environment is activated: `source .venv/bin/activate`

## License

USE IT TO BE FREE. USE IT TO WHAT YOU WANT ! Free i don't need your money
