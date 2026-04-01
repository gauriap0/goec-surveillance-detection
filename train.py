from roboflow import Roboflow
import torch
from ultralytics import YOLO
import os

# Download dataset
rf = Roboflow(api_key="9kYAcrA7GzElMnTccH01")
project = rf.workspace("annotation-0pigh").project("goec")
version = project.version(4)
dataset = version.download("yolov8")

# Detect device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# Train
model = YOLO("yolov8n.pt")
model.train(
    data="combined/data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device=device,
    project="runs/train",
    name="goec_v4",
    save=True,
)

print("Training complete!")
print("Best weights: runs/train/goec_v4/weights/best.pt")
