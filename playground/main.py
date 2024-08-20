import json
import os
import cv2
import mediapipe as mp


def append_points_to_json_file(data, filename="points.json"):
    # Check if file exists and is not empty
    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0

    if not file_exists:
        with open(filename, "w", encoding="utf-8") as file:
            # File does not exist or is empty, start a new JSON array
            file.write("[")
            json_data = json.dumps(data, indent=4)
            print(json_data)
            file.write(json_data)
            file.write("]")
    else:
        with open(filename, "r+", encoding="utf-8") as fh:
            # goto to the end of file
            fh.seek(0, os.SEEK_END)
            # remove the last character, open the JSON array
            fh.seek(fh.tell() - 1)
            fh.write(",")

        with open(filename, "a", encoding="utf-8") as file:
            # Convert and append each point, except the last, with a comma
            json_data = json.dumps(data, indent=4)
            print(json_data)
            file.write(json_data)
            # Close the JSON array
            file.write("]")


path = "gymvideo.mp4"  # "../videos/WhatsApp Video 2024-04-24 at 19.56.05.mp4"
print("setting up mediapipe")
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

print("loading")
cap = cv2.VideoCapture(path)
up = False
counter = 0
print("loaded")
while True:
    success, img = cap.read()
    # img = cv2.resize(img, (1280,720))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    # print("-----------------------------------------------------")
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        points = {}
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id,lm,cx,cy)
            points[id] = (cx, cy)
        print(points)
        append_points_to_json_file(points, "points-gym.json")
        # cv2.circle(img, points[12], 15, (255,0,0), cv2.FILLED)
        # cv2.circle(img, points[14], 15, (255,0,0), cv2.FILLED)
        # cv2.circle(img, points[11], 15, (255,0,0), cv2.FILLED)
        # cv2.circle(img, points[13], 15, (255,0,0), cv2.FILLED)

        if not up and points[14][1] + 40 < points[12][1]:
            print("UP")
            up = True
            counter += 1
        elif points[14][1] > points[12][1]:
            print("Down")
            up = False
        # print("----------------------",counter)

    cv2.putText(
        img, str(counter), (100, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 12
    )

    cv2.imshow("img", img)
    key = cv2.waitKey(0)
    if key == ord("q"):
        print("Quitting")
        break
