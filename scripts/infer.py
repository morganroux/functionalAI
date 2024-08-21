import sys
import numpy as np
import torch
from train import WorkoutClassifier
from transform import process_video
from dataset import get_classes

import warnings
warnings.filterwarnings("ignore")

class_names, n_classes = get_classes()
# class KeypointsDataset(Dataset):
#     def __init__(self, csv_file, label, group_size=30, overlap=20):
#         self.csv_file = csv_file
#         self.group_size = group_size
#         self.overlap = overlap
#         self.step = group_size - overlap
#         self.data = pd.read_csv(csv_file)
#         columns = ["frame"]
#         indices = np.concatenate([np.array([0]), np.arange(12, 21), np.arange(24, 33)])
#         for i in indices:
#             columns.extend([f"{i}_x", f"{i}_y", f"{i}_z"])
#         self.data = self.data[columns]
#         self.label = label
#         self.length = (len(self.data) - group_size) // self.step + 1

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         start_idx = idx * self.step
#         end_idx = start_idx + self.group_size
#         group = [
#             torch.tensor(self.data.iloc[i].values, dtype=torch.float32)
#             for i in range(start_idx, end_idx)
#         ]
#         keypoints = torch.stack(group)
#         return keypoints, torch.tensor(self.label), self.csv_file


def load_keypoints(data, group_size, step):

    # Select relevant columns
    columns = ["frame"]
    indices = np.concatenate([np.array([0]), np.arange(12, 21), np.arange(24, 33)])
    for i in indices:
        columns.extend([f"{i}_x", f"{i}_y", f"{i}_z"])
    data = data[columns]

    # Calculate the number of groups
    length = (len(data) - group_size) // step + 1

    # Break data into smaller groups
    groups = []
    for idx in range(length):
        start_idx = idx * step
        end_idx = start_idx + group_size
        group = [
            torch.tensor(data.iloc[i].values, dtype=torch.float32)
            for i in range(start_idx, end_idx)
        ]
        keypoints = torch.stack(group)
        groups.append(keypoints)

    return torch.stack(groups)


def infer(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        print([class_names.iloc[pred]["class_name"] for pred in predicted.tolist()])
        mode_value, _ = torch.mode(predicted)
        return predicted, mode_value.item()


if len(sys.argv) > 1:
    filename = sys.argv[1]
    df = process_video(filename)
    group_size = 30  # Example group size
    step = 10  # Example step size
    keypoint_groups = load_keypoints(df, group_size, step)
    print(keypoint_groups.size())
    print(f"Loaded {len(keypoint_groups)} groups of keypoints.")
    print(f"First group shape: {keypoint_groups[0].shape}")
    print("----------")
    # Load the model
    model = torch.load("runs/crossfit_6/models/model.pth", weights_only=False)
    _, mode = infer(model, keypoint_groups)
    print("\nFinal prediction:", class_names.iloc[mode]["class_name"])

else:
    print("No filename provided.")
