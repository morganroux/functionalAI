import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class FileDataset(Dataset):
    def __init__(self, csv_files, labels):
        if len(csv_files) != len(labels):
            raise ValueError("csv_files and labels must have the same length")
        self.data = csv_files
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    file_datasets = []
    class_names = []
    for folder in sorted(os.listdir("./keypoints")):
        if not os.path.isdir(f"./keypoints/{folder}"):
            continue
        class_names.append(folder)
        print(f"folder: {folder}")
        files = sorted(os.listdir(f"./keypoints/{folder}"))
        label = len(class_names) - 1
        dataset = FileDataset(
            [f"./keypoints/{folder}/{file}" for file in files],
            [label for file in files],
        )
        print(f"   {len(dataset)} files")
        file_datasets.append(dataset)

    n_classes = len(class_names)
    file_dataset = torch.utils.data.ConcatDataset(file_datasets)

    train_size = int(0.8 * len(file_dataset))
    val_size = len(file_dataset) - train_size
    train_filedataset, val_filedataset = torch.utils.data.random_split(
        file_dataset, [train_size, val_size]
    )

    data_list = []
    for filename, label in train_filedataset:
        class_name = class_names[label]
        data_list.append([filename, label, class_name])

    df_train = pd.DataFrame(data_list)

    data_list = []
    for filename, label in val_filedataset:
        class_name = class_names[label]
        data_list.append([filename, label, class_name])
    df_val = pd.DataFrame(data_list)

    # Optionally, you can name the columns if you know the size of your data
    # For example, if you're working with MNIST images (28x28 pixels) plus a label
    columns = ["filename", "label", "class_name"]
    df_train.columns = columns
    df_val.columns = columns
    df_train.to_csv("keypoints/train_filedataset.csv", index=False)
    df_val.to_csv("keypoints/val_filedataset.csv", index=False)

    df_classes = pd.DataFrame(class_names, columns=["class_name"])
    df_classes.to_csv("keypoints/classes.csv", index=False)
