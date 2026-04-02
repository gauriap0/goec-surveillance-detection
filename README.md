GOEC Surveillance Detection

YOLOv8 object detection model for GOEC charging station surveillance.

 Detects
- Cars
- Gun chargers
- Fire
- Smoke

 Results
- Overall mAP50: 0.626
- Gun detection: 0.805
- Car detection: 0.726

 Training
- Dataset: 6,237 images (3 Roboflow datasets merged)
- Model: YOLOv8n
- Epochs: 50
- GPU: Google Colab T4
