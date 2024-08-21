import torch
from train import WorkoutClassifier
from dataset import n_classes

checkpoint = torch.load("runs/crossfit_5/checkpoints/checkpoint_epoch_50.pth")
model = WorkoutClassifier(num_features=58, num_classes=n_classes)
model.load_state_dict(checkpoint)
torch.save(model, "runs/crossfit_5/models/model.pth")
