from ultralytics import YOLO


def train_model():
    model = YOLO("yolov8s.pt")
    model.train(
        data="drone.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=0,
        device=0
    )


if __name__ == "__main__":
    train_model()
