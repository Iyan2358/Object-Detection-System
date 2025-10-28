
# Real-Time Object Detection with YOLOv8 + Webcam

This project uses a webcam feed and a pretrained **YOLOv8** model to detect objects from the COCO(Common Objects in Context) dataset is a large-scale object detection, segmentation, and captioning dataset. The detections are filtered to only show specific classes such as person, car, dog, laptop, bicycle, etc.

## Features

• Real-time detection from webcam
• Pretrained model (no training required)
• Filters detected objects to a custom list
• Bounding boxes and confidence scores displayed live

## Requirements

• Python 3.8+
• Webcam
• GPU recommended (RTX series works well)

## Installation

Install the required Python packages:

```bash
pip install ultralytics
pip install opencv-python
```

## Running the Script

Run the detection script:

```bash
python webcam_detect.py
```

A window will open and show live detections from your webcam.
Press **Q** to exit.

## How It Works

1. Loads a pretrained YOLOv8 model trained on (COCO)[https://cocodataset.org] classes.
2. Captures frames from the webcam.
3. Runs inference on each frame.
4. Filters results to only show predefined object classes.
5. Draws bounding boxes with labels and confidence.

## Customizing Classes

Edit the `allowed_classes` list inside the script to include only the objects you want to detect.

## Notes

• GPU acceleration improves FPS significantly.
• Increase `conf=0.45` for stricter detection, lower it to be more permissive.
• Works with any YOLOv8 model variant (`n`, `s`, `m`, `l`, `x`).


