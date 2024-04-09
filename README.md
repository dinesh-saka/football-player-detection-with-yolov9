# Football Player Detection With YOLOv9

## Overview

This project aims to detect football players in images using the YOLOv9 object detection framework. YOLO (You Only Look Once) is a state-of-the-art real-time object detection system that can detect multiple objects in an image simultaneously with high accuracy.

## Setup

### Requirements

- Python 3.x
- NVIDIA GPU with CUDA support for faster training (recommended)
- NVIDIA GPU drivers
- CUDA Toolkit
- cuDNN library

### Installation

1. Clone the YOLOv9 repository:

   ```bash
   git clone https://github.com/SkalskiP/yolov9.git
   cd yolov9
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the Roboflow library:

   ```bash
   pip install roboflow
   ```

4. Download the pre-trained model weights:

   ```bash
   wget -P /path/to/weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
   wget -P /path/to/weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
   ```

5. Authenticate and download the dataset using Roboflow:

   ```python
   roboflow.login()
   rf = roboflow.Roboflow()
   project = rf.workspace("your_workspace").project("your_project")
   version = project.version(version_number)
   dataset = version.download("yolov9")
   ```

## Usage

### Training

To train the custom model, run the following command:

```bash
python train.py \
--batch 16 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data /path/to/dataset/data.yaml \
--weights /path/to/weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
```

### Validation

To validate the custom model, run:

```bash
python val.py \
--img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 \
--data /path/to/dataset/data.yaml \
--weights /path/to/yolov9/runs/train/exp/weights/best.pt
```

## Results

You can find the results of the training and validation in the `runs/train/exp/` directory. The `results.png` file contains the precision, recall, and F1 score curves, while the `confusion_matrix.png` file displays the confusion matrix. Sample predictions can be found in the `runs/detect/exp/` directory.
