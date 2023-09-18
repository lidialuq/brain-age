import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
from transforms import train_transforms, val_transforms
from dataset import UKB
from trainer import Trainer
from model import SFCN, get_Bb_dims
import os
import sys
sys.path.append('../')
from config import get_paths

# 1. Define dataset path and hyperparameters
device = torch.device("cuda:0")
config = {
   "learning_rate": 0.001,
   "weight_decay": 5e-4,
   "momentum": 0.9,
   "batch_size": 32,
   "epochs": 1000,
   "data_augmentation": 1.5,
   "architecture": "sfcn",
   "model_depth": 5,
}
# 2. Initialize wandb
wandb.init(
   project="brain-age",
   config=config,
)
# 3. Initialize datasets and dataloaders
data_path, masks_path = get_paths()
train_dataset = UKB(data_path, masks_path, train=True, val_split=0.2,
       transforms=train_transforms)
val_dataset = UKB(data_path, masks_path, train=False, val_split=0.2,
       transforms=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
       shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
       shuffle=False)
# 4. Setup model, loss, optimizer, and trainer
model_dims = get_Bb_dims(config["model_depth"])
model = SFCN(model_dims, 1).to(device)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
         momentum=config["momentum"], weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
       config["epochs"])
trainer = Trainer(model, train_loader, val_loader, criterion,
       optimizer, scheduler, device=device)
# 5. Train the model
trainer.train(epochs=config["epochs"])







