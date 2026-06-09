from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('weights/best.pt')

    results = model(source=0, verbose=True, imgsz=736, stream=False, show=True, save=False, save_txt=False)
