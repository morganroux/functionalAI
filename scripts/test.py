import torch
from torch.utils.data import DataLoader
from train import WorkoutClassifier
from dataset import val_dataset, class_names

model = torch.load("runs/crossfit_6/models/model.pth", weights_only=False)

val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
data_iter = iter(val_loader)
first_batch = next(data_iter)

inputs, labels, files = first_batch
outputs = model(inputs)
_, predicted = torch.max(outputs, 1)
# print(inputs, inputs.shape)
# print("outputs: ", outputs)
print(
    f"\nFiles: {list(files)} \nPredicted: {[class_names.iloc[pred]['class_name'] for pred in predicted.tolist()]}  \nActual: {[class_names.iloc[label]['class_name'] for label in labels.tolist()]}"
)
