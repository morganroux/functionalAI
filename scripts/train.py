import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class WorkoutClassifier(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        kernel_size=5,
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
    def __init__(self, csv_file, label, group_size=30, overlap=25):
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
        return keypoints, torch.tensor(self.label)


def train_model(
    model, train_loader, criterion, optimizer, num_epochs=25, run="crossfit"
):
    writer = SummaryWriter(f"runs/{run}")

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
            print(
                f"Batch [{batch_number+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

        # Print statistics at the end of the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )
        if writer is not None:
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_acc, epoch)

    print("Training complete")
    writer.close()


datasetAndy = KeypointsDataset("./keypoints/andy_keypoints.csv", 0)
datasetGym = KeypointsDataset("./keypoints/gymvideo_keypoints.csv", 1)
dataset = torch.utils.data.ConcatDataset([datasetAndy, datasetGym])

print(f"Total data points: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = WorkoutClassifier(num_features=58, num_classes=2)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, dataloader, criterion, optimizer, num_epochs=25)
torch.save(model, "models/model.pth")
