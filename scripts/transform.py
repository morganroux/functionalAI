import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to process a video and extract keypoints
def process_video(video_path, output_csv_path):
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
    columns = ['frame'] + [f'{name}_{axis}' for name in range(33) for axis in ['x', 'y', 'z', 'visibility']]
    keypoints_df = pd.DataFrame(keypoints_list, columns=columns)
    keypoints_df.to_csv(output_csv_path, index=False)

    cap.release()
    print(f"Finished processing {video_path} and saved keypoints to {output_csv_path}")

# Process all videos in a directory
def process_videos_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4") or filename.endswith(".avi"):  # Adjust file extensions as necessary
            video_path = os.path.join(input_dir, filename)
            output_csv_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_keypoints.csv")
            process_video(video_path, output_csv_path)

# Example usage
input_video_directory = "./videos"
output_keypoints_directory = "./keypoints"

process_videos_in_directory(input_video_directory, output_keypoints_directory)
