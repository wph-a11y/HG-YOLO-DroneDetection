from ultralytics import YOLO
import cv2


def main():
    model = YOLO("weights/best.pt")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=0.5,
            imgsz=640,
            half=True,
            device='cuda:0',
            verbose=False
        )

        if isinstance(results, list):
            annotated_frame = results[0].plot()
        else:
            annotated_frame = next(results).plot()

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Drone Detection", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
