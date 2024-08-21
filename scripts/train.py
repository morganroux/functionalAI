import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix

from dataset import get_datasets, get_classes

class_names, n_classes = get_classes()


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


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10, run="crossfit"
):
    run_number = 1
    while os.path.exists(f"runs/{run}_{run_number}"):
        run_number += 1
    run = f"{run}_{run_number}"
    writer = SummaryWriter(f"runs/{run}")

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels, csv_files in train_loader:

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
        val_loss, accuracy, precision, recall, f1, cm = eval_model(
            model, val_loader, criterion
        )

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            if not os.path.exists(f"runs/{run}/checkpoints"):
                os.makedirs(f"runs/{run}/checkpoints")
            checkpoint_path = f"runs/{run}/checkpoints/checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        print("------------\n")
        if writer is not None:
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", accuracy, epoch)
            writer.add_scalar("Precision/val", precision, epoch)
            writer.add_scalar("Recall/val", recall, epoch)
            writer.add_scalar("F1/val", f1, epoch)

    print("Training complete")
    writer.close()
    if not os.path.exists(f"runs/{run}/models"):
        os.makedirs(f"runs/{run}/models")
    torch.save(model, f"runs/{run}/models/model.pth")
    print(f"Model saved at runs/{run}/models/model.pth")


def eval_model(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels, csv_files in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted"
    )
    cm = confusion_matrix(true_labels, predicted_labels)

    print(f"Validation :")
    print(
        f"Loss: {val_loss:.4f}, Validation: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return val_loss, accuracy, precision, recall, f1, cm


model = WorkoutClassifier(num_features=58, num_classes=n_classes)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# dont run on import
if __name__ == "__main__":
    train_dataset, val_dataset = get_datasets()
    # datasets = []
    # for folder in sorted(os.listdir("./keypoints")):
    #     if not os.path.isdir(f"./keypoints/{folder}"):
    #         continue
    #     print(f"folder: {folder}")
    #     files = sorted(os.listdir(f"./keypoints/{folder}"))
    #     if len(files) > 0:
    #         label = len(datasets)
    #         folder_datasets = []
    #         for file in files:
    #             ds = KeypointsDataset(f"./keypoints/{folder}/{file}", label)
    #             try:
    #                 len(ds)
    #             except:
    #                 continue
    #             folder_datasets.append(ds)
    #             # print(f"  {file}")
    #             # print(f" {len(ds)} data points")

    #         dataset = torch.utils.data.ConcatDataset(folder_datasets)
    #         print(f"   {len(folder_datasets)} files - {len(dataset)} data points")
    #         datasets.append(dataset)

    # # datasetAndy = KeypointsDataset(["./keypoints/andy_keypoints.csv"], 0)
    # # datasetGym = KeypointsDataset(["./keypoints/gymvideo_keypoints.csv"], 1)
    # dataset = torch.utils.data.ConcatDataset(datasets)
    # print("----------")
    # print(f"Total data points: {len(dataset)}")
    # # number of classes
    # n_classes = len(datasets)
    # print(f"Number of classes: {n_classes}")
    # print("----------\n")

    # # Split train and validation dataset
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, val_size]
    # )

    # # Create data loaders for train and validation dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Training model...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

    # eval_model(model, val_loader, criterion)
