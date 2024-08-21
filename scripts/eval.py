
from train import eval_model, criterion, WorkoutClassifier
from dataset import val_dataset, class_names
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model = torch.load("runs/crossfit_6/models/model.pth", weights_only=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

val_loss, accuracy, precision, recall, f1, cm = eval_model(model, val_loader, criterion)

# print("Confusion Matrix:")
# print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(class_names), yticklabels=np.unique(class_names))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
