import cv2
import os


def extract_frames(video_path, output_folder, every_n_frames=10):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % every_n_frames == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        # Print progress every 50 frames (avoid spam)
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.2f}% ({frame_count}/{total_frames})", end="\r")

        frame_count += 1

    cap.release()

    print("\nDone.")
    print(f"Extracted {saved_count} frames to '{output_folder}'")


# Example usage
if __name__ == "__main__":
    video_path = "input.mp4"
    output_folder = "./Auxiliary_Dataset/inputmp4/"
    extract_frames(video_path, output_folder, every_n_frames=1)
    video_path = "Input sample.mp4"
    output_folder = "./Auxiliary_Dataset/input_sample/"
    extract_frames(video_path, output_folder, every_n_frames=1)