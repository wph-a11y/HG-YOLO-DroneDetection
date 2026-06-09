# HG-YOLO: Low-Altitude Security UAV Detection System

This project is a course practice assignment based on an undergraduate thesis in Control Science and Engineering. It implements a UAV (drone) detection system using improved YOLO architecture, originally developed as the perception module of a robotic visual servoing control system, and later improved for low-altitude security applications.

> **Note**: This is not a professional research product in the field of low-altitude security or drone detection. The primary research direction of the author is robotic visual servoing control, and drone detection is an application scenario of the perception module. The code and data are provided for learning and reference purposes only.

## Features

- **Improved YOLO Architecture**: Custom C3k2 module for enhanced feature extraction
- **Real-time Detection GUI**: PyQt5-based monitoring interface with alarm system
- **Multi-source Input**: Supports live camera, image, and video file detection
- **Drone Classification**: Detects and classifies multiple drone types (drone_1, drone_2, drone_3, drone_others)
- **Data Collection Tools**: Bing image scraper and video frame extractor for dataset building
- **Visualization Tools**: Literature statistics visualization and mIoU analysis

## Project Structure

```
HG-YOLO-DroneDetection/
├── drone.yaml              # Dataset configuration
├── custom_modules.py       # Custom C3k2 module
├── train.py                # Model training script
├── detect_system.py        # Main detection GUI (Drone Defense System)
├── system.py               # Generic YOLO detection GUI
├── infer.py                # Basic inference script
├── infer3.py               # Inference with custom module support
├── data_collect.py         # Drone image scraper (Bing)
├── frame_extract.py        # Video frame extractor
├── draw.py                 # Literature statistics visualization
├── draw2.py                # Literature statistics visualization (v2)
├── mIoU-k.py               # mIoU analysis with k-means clustering
├── weights/                # Model weights directory
└── drone_images/           # Sample drone images
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

Edit `train.py` to adjust training parameters (epochs, batch size, image size, etc.).

### Detection GUI

```bash
python detect_system.py
```

The GUI supports:
- Live camera detection
- Image file detection
- Video file detection
- Adjustable confidence and IOU thresholds
- Real-time alarm when drone is detected
- Snapshot saving

### Basic Inference

```bash
python infer.py
```

### Data Collection

```bash
# Scrape drone images from Bing
python data_collect.py

# Extract frames from drone videos
python frame_extract.py
```

## Dataset

The `drone.yaml` configuration defines 4 drone classes:
- drone_1
- drone_2
- drone_3
- drone_others

Prepare your dataset following the YOLO format and update the paths in `drone.yaml`.

## Suggested Research Directions

For researchers interested in low-altitude security UAV detection, the following directions may be worth exploring:

1. **Detection-Tracking-Control Co-design**: Closing the loop between detection and pan-tilt/sensor control for active perception and adaptive search.
2. **Multi-Object Data Association**: Addressing the core challenge of MOT (Multi-Object Tracking) in drone swarm scenarios using Kalman filtering + Hungarian algorithm, with Re-ID assisted association.
3. **Edge Deployment & Real-time Performance**: Model lightweighting (knowledge distillation, pruning, quantization) and edge computing deployment (TensorRT acceleration).
4. **Cross-domain Transfer & Domain Adaptation**: Improving generalization under varying urban conditions (lighting, weather, occlusion).

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for detailed dependencies

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{10.1145/3807246.3807363,
author = {Wang, Puhui and Li, Chuanyang and Zhang, Jiaqi and Pang, Yuanlong and Zhang, Xiangrui and Wan, Xinyi and Hu, Changhua},
title = {Low-Altitude Security UAV Detection: Fusing Generative Augmentation with Hypergraph Computing},
year = {2026},
isbn = {9798400723537},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3807246.3807363},
doi = {10.1145/3807246.3807363},
booktitle = {Proceedings of the 2026 International Conference on Artificial Intelligence and Control},
pages = {744–751},
numpages = {8},
series = {CAIC '26}
}

```

## License

This project is for academic research purposes only. Commercial use is prohibited.
