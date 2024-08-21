import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


# Define a custom dataset class
class KeypointsDataset(Dataset):
    def __init__(self, csv_file, label, group_size=30, overlap=20):
        self.csv_file = csv_file
        self.group_size = group_size
        self.overlap = overlap
        self.step = group_size - overlap
        self.data = pd.read_csv(csv_file)
        columns = ["frame"]
        indices = np.concatenate([np.array([0]), np.arange(12, 21), np.arange(24, 33)])
        for i in indices:
            columns.extend([f"{i}_x", f"{i}_y", f"{i}_z"])
        self.data = self.data[columns]
        self.label = label
        self.length = (len(self.data) - group_size) // self.step + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx * self.step
        end_idx = start_idx + self.group_size
        group = [
            torch.tensor(self.data.iloc[i].values, dtype=torch.float32)
            for i in range(start_idx, end_idx)
        ]
        keypoints = torch.stack(group)
        return keypoints, torch.tensor(self.label), self.csv_file


# file_datasets = []
# class_names = []
# for folder in sorted(os.listdir("./keypoints")):
#     if not os.path.isdir(f"./keypoints/{folder}"):
#         continue
#     class_names.append(folder)
#     print(f"folder: {folder}")
#     files = sorted(os.listdir(f"./keypoints/{folder}"))
#     label = len(class_names) - 1
#     dataset = FileDataset(
#         [f"./keypoints/{folder}/{file}" for file in files], [label for file in files]
#     )
#     print(f"   {len(dataset)} files")
#     file_datasets.append(dataset)

# n_classes = len(class_names)
# file_dataset = torch.utils.data.ConcatDataset(file_datasets)

# train_size = int(0.8 * len(file_dataset))
# val_size = len(file_dataset) - train_size
# train_filedataset, val_filedataset = torch.utils.data.random_split(
#     file_dataset, [train_size, val_size]
# )
class_names  = pd.read_csv("./keypoints/classes.csv")
n_classes = len(class_names)

train_filedataset = pd.read_csv("./keypoints/train_filedataset.csv")
val_filedataset = pd.read_csv("./keypoints/val_filedataset.csv")

train_dataset = []
for index, row in train_filedataset.iterrows():
    ds = KeypointsDataset(row['filename'], row['label'])
    try:
        len(ds)
    except:
        continue
    train_dataset.append(ds)

train_dataset = torch.utils.data.ConcatDataset(train_dataset)


val_dataset = []
for index, row in val_filedataset.iterrows():
    ds = KeypointsDataset(row['filename'], row['label'])
    try:
        len(ds)
    except:
        continue
    val_dataset.append(ds)

val_dataset = torch.utils.data.ConcatDataset(val_dataset)

print("----------")
print(f"Total train data points: {len(train_dataset)}")
print(f"Total val data points: {len(val_dataset)}")
print(f"Number of classes: {n_classes}")

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
# iterloader = iter(train_loader)
# inputs, labels, csv_file = next(iterloader)
# print(inputs.size(), [class_names[label] for label in labels], csv_file)
