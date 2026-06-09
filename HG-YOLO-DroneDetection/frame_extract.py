import cv2
import os
import numpy as np


def extract_frames_by_interval(video_path, output_dir, frame_interval=2):
    if frame_interval <= 0:
        print("Error: frame interval must be a positive integer!")
        return

    os.makedirs(output_dir, exist_ok=True)

    def cv2_imread_chinese(path):
        stream = cv2.VideoCapture(path.encode('utf-8').decode('gbk'))
        return stream

    def cv2_imwrite_chinese(save_path, frame):
        dir_name, file_name = os.path.split(save_path)
        temp_file = os.path.join(dir_name, f"temp_{np.random.randint(10000)}.jpg")
        success = cv2.imwrite(temp_file, frame)
        if success:
            os.rename(temp_file, save_path)
        return success

    cap = cv2_imread_chinese(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"===== Video Info =====")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration // 60:.0f}m{duration % 60:.1f}s")
    print(f"Extract interval: every {frame_interval} frames")
    print(f"Estimated output: {total_frames // frame_interval} frames")
    print("====================")

    frame_index = 0
    saved_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if success:
            if frame is None or frame.size == 0:
                frame_index += 1
                continue

            if frame_index % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f"drone_1_{saved_count:04d}.jpg")
                save_success = cv2_imwrite_chinese(frame_filename, frame)
                if save_success:
                    saved_count += 1
                    if saved_count % 50 == 0:
                        progress = (frame_index / total_frames) * 100
                        print(f"Progress: saved {saved_count} frames | {progress:.1f}%")
            frame_index += 1

    cap.release()

    final_progress = (saved_count / (total_frames // frame_interval)) * 100 if total_frames > 0 else 0
    print(f"\n===== Extraction Complete =====")
    print(f"Saved frames: {saved_count}")
    print(f"Output directory: {output_dir}")
    print(f"Completion rate: {final_progress:.1f}%")


if __name__ == "__main__":
    VIDEO_PATH = "drone_video.mp4"
    OUTPUT_DIR = "drone_frames"
    FRAME_INTERVAL = 2

    try:
        extract_frames_by_interval(VIDEO_PATH, OUTPUT_DIR, FRAME_INTERVAL)
    except Exception as e:
        print(f"Error: {type(e).__name__} - {e}")
