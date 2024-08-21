import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)


# Function to process a video and extract keypoints
def process_video(video_path, output_csv_path=None):
    print(f"Processing video: {video_path}")
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_number = 0
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and extract keypoints
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append(landmark.x)
                keypoints.append(landmark.y)
                keypoints.append(landmark.z)
                keypoints.append(landmark.visibility)

            keypoints_list.append([frame_number] + keypoints)

        frame_number += 1

    # Convert to a DataFrame and save as CSV
    columns = ["frame"] + [
        f"{name}_{axis}" for name in range(33) for axis in ["x", "y", "z", "visibility"]
    ]
    keypoints_df = pd.DataFrame(keypoints_list, columns=columns)
    print(f"Finished processing {video_path}, extracted {len(keypoints_df)} keypoints")
    if output_csv_path is not None:
        keypoints_df.to_csv(output_csv_path, index=False)
        print(f"Saved keypoints to {output_csv_path}")
    cap.release()
    return keypoints_df


# Process all videos in a directory
def process_videos_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in sorted(os.walk(input_dir)):
        print(f"========= Processing videos in {root} ============")
        if root == input_dir or root.startswith(input_dir + os.path.sep):
            for filename in sorted(files):
                if filename.endswith(".mp4") or filename.endswith(".avi"):
                    # Adjust file extensions as necessary
                    video_path = os.path.join(root, filename)
                    output_subdir = os.path.join(
                        output_dir, os.path.relpath(root, input_dir)
                    )
                    os.makedirs(output_subdir, exist_ok=True)
                    output_csv_path = os.path.join(
                        output_subdir, f"{os.path.splitext(filename)[0]}_keypoints.csv"
                    )
                    process_video(video_path, output_csv_path)


if __name__ == "__main__":
    input_video_directory = "./videos/workout"
    output_keypoints_directory = "./keypoints"

    process_videos_in_directory(input_video_directory, output_keypoints_directory)
