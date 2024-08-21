import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from train import WorkoutClassifier


csv_file = "./keypoints/andy_keypoints.csv"
classes = []
for folder in sorted(os.listdir("./keypoints")):
    if not os.path.isdir(f"./keypoints/{folder}"):
        continue
    # print(f"folder: {folder}")
    classes.append(folder)

group_size = 30
overlap = 20
step = group_size - overlap

data = pd.read_csv(csv_file)
columns = ["frame"]
indices = np.concatenate([np.array([0]), np.arange(12, 21), np.arange(24, 33)])
for i in indices:
    columns.extend([f"{i}_x", f"{i}_y", f"{i}_z"])
data = data[columns]
length = (len(data) - group_size) // step + 1

idx = 30
start_idx = idx * step
end_idx = start_idx + group_size
group = [
    torch.tensor(data.iloc[i].values, dtype=torch.float32)
    for i in range(start_idx, end_idx)
]
keypoints = torch.stack(group)
keypoints = keypoints.unsqueeze(0)
inputs = keypoints  # torch.tensor(self.label), self.csv_file

print(inputs, inputs.shape)


model = torch.load("models/model.pth", weights_only=False)

outputs = model(inputs)
_, predicted = torch.max(outputs, 1)
probabilities = F.softmax(outputs, dim=1)
print("outputs: ", outputs)
print("probabilities: ", probabilities)
print("predicted ", predicted, [classes[pred] for pred in predicted])
# print(f"Files: {list(files)} \nPredicted: {[classes[pred] for pred in predicted]} \nActual: {[classes[label] for label in labels]}")
