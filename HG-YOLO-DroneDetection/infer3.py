import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, C3, Bottleneck, C2f, Detect
from custom_modules import C3k2

torch.serialization.add_safe_globals([
    DetectionModel, Conv, C3, C3k2,
    Bottleneck, C2f, Detect
])

if __name__ == "__main__":
    model = YOLO('weights/best.pt')

    results = model.predict(
        source=0,
        stream=True,
        show=True,
        imgsz=640,
        conf=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        print(f"Detected {len(boxes)} targets")
