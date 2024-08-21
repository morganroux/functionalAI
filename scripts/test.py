import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class WorkoutClassifier(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        kernel_size=9,
        lstm_hidden_size=64,
        lstm_layers=1,
        dropout_rate=0.5,
    ):
        super(WorkoutClassifier, self).__init__()

        # 1D Convolutional Layer
        self.conv1 = nn.Conv1d(
            in_channels=num_features, out_channels=64, kernel_size=kernel_size
        )

        # Pooling Layer
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Dropout Layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Fully Connected Layer (Output Layer)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # x is expected to have shape (batch_size, num_frames, num_features)

        # Permute to match Conv1d input (batch_size, num_features, num_frames)
        x = x.permute(0, 2, 1)

        # Apply Conv1d -> ReLU -> MaxPool1d -> Dropout
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Permute to match LSTM input (batch_size, num_frames_after_pooling, 64)
        x = x.permute(0, 2, 1)

        # Apply LSTM
        x, _ = self.lstm(x)

        # Take the output from the last time step
        x = x[:, -1, :]

        # Apply Fully Connected Layer (for classification)
        x = self.fc(x)

        return x


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


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10, run="crossfit"
):
    run_number = 1
    while os.path.exists(f"runs/tensorboard/{run}_{run_number}"):
        run_number += 1
    run = f"{run}_{run_number}"
    writer = SummaryWriter(f"runs/tensorboard/{run}")

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_number, (inputs, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)  # BCEWithLogitsLoss expects float labels
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(
            #     f"Batch [{batch_number+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            # )

        # Print statistics at the end of the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print("")
        print(
            f"Epoch [{epoch+1}/{num_epochs}] \nTraining Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}"
        )
        val_loss, val_acc = eval_model(model, val_loader, criterion)

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            if not os.path.exists(f"runs/checkpoints/{run}"):
                os.makedirs(f"runs/checkpoints/{run}")
            checkpoint_path = f"runs/checkpoints/{run}/checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        print("------------\n")
        if writer is not None:
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

    print("Training complete")
    writer.close()


def eval_model(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    return val_loss, val_acc


# datasetAndy = KeypointsDataset(["./keypoints/andy_keypoints.csv"], 0)
# datasetGym = KeypointsDataset(["./keypoints/gymvideo_keypoints.csv"], 1)

datasets = []
classes = []
for folder in sorted(os.listdir("./keypoints")):
    if not os.path.isdir(f"./keypoints/{folder}"):
        continue
    # print(f"folder: {folder}")
    classes.append(folder)
    files = sorted(os.listdir(f"./keypoints/{folder}"))
    if len(files) > 0:
        label = len(datasets)
        folder_datasets = []
        for file in files:
            ds = KeypointsDataset(f"./keypoints/{folder}/{file}", label)
            try:
                len(ds)
            except:
                continue
            folder_datasets.append(ds)
            # print(f"  {file}")
            # print(f" {len(ds)} data points")

        dataset = torch.utils.data.ConcatDataset(folder_datasets)
        # print(f"   {len(folder_datasets)} files - {len(dataset)} data points")
        datasets.append(dataset)

dataset = torch.utils.data.ConcatDataset(datasets)
print("----------")
print(f"Total data points: {len(dataset)}")
# number of classes
n_classes = len(datasets)
print(f"Number of classes: {n_classes}")
for class_name, idx in enumerate(classes):
    print(class_name, idx)
print("----------\n")


# Split train and validation dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Create data loaders for train and validation dataset
val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

model = torch.load("models/model.pth", weights_only=False)

data_iter = iter(val_loader)
first_batch = next(data_iter)

# If the DataLoader returns a tuple (inputs, labels), you can unpack it
inputs, labels, files = first_batch
print(inputs, inputs.shape)
outputs = model(inputs)
_, predicted = torch.max(outputs, 1)
print("outputs: ", outputs)
print(f"Files: {list(files)} \nPredicted: {[classes[pred] for pred in predicted]} \nActual: {[classes[label] for label in labels]}")
