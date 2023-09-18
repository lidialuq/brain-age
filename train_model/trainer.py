import torch
from torch import nn
import wandb
import os
from tqdm import tqdm
import random
import sys
sys.path.append('../')
from config import PROJECT_ROOT

class Trainer:
    def __init__(self, model, dataset, train_loader, val_loader, criterion, 
                optimizer, scheduler, device="cuda"):
        self.model = model.to(device)
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
        if wandb.run.name: # If wnadb online
            self.save_model_folder = os.path.join(PROJECT_ROOT, 'trained_models', 
                    wandb.run.name)
        else:
            self.save_model_folder = os.path.join(PROJECT_ROOT, 'trained_models', 
                    str(random.randint(0, 1000000)))

        wandb.watch(self.model)


    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        age_deltas = []
        for image, label in tqdm(self.train_loader): 
            inputs, targets = image.float().to(self.device), label.float().to(self.device)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            self.scaler.scale(loss).backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            # convert output back to age and calculate difference to target age
            age_predicted = (outputs * self.dataset.std_age) + self.dataset.mean_age
            age_target = (targets * self.dataset.std_age) + self.dataset.mean_age
            print(age_predicted)
            age_delta = age_predicted - age_target
            age_deltas.extend(age_delta)
        self.scheduler.step()
        avg_loss = total_loss / len(self.train_loader)
        age_delta_mean = torch.mean(torch.stack(age_deltas))
        return avg_loss, age_delta_mean

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        with torch.no_grad():
            for image, label in self.val_loader:
                inputs, targets = image.float().to(self.device), label.float().to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct_predictions / len(self.val_loader.dataset)
        return avg_loss, accuracy

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_delta = self.train_one_epoch()
            val_loss, val_delta = self.validate()

            print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train delta: {train_delta:.2f}%, Val Loss: {val_loss:.4f}, Val delta: {val_delta:.2f}%")
            wandb.log({"train": {"loss": train_loss, "delta": train_delta}, "val": {"loss": val_loss, "delta": val_delta}})
            # Save the model every 1 epochs
            if (epoch+1) % 1 == 0:
                if not os.path.exists(self.save_model_folder):
                    os.makedirs(self.save_model_folder)
                self.save_model(os.path.join(self.save_model_folder,
                        f"model_epoch_{epoch+1}.pth"))
        self.save_model(os.path.join(self.save_model_folder, "model_final.pth"))
        wandb.finish()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)