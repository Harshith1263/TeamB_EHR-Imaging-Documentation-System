import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from srcnn_architecture import SRCNN 

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MILESTONE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
TRAIN_DIR = os.path.join(MILESTONE_DIR, "train")
HR_DIR = os.path.join(TRAIN_DIR, "HR")
LR_DIR = os.path.join(TRAIN_DIR, "LR")
MODEL_DIR = os.path.join(MILESTONE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        if self.transform:
            hr = self.transform(hr)
            lr = self.transform(lr)

        return lr, hr

# Training
def train_model(epochs=20, batch_size=16, lr_rate=1e-4, resize_dim=128):
    transform = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor()
    ])

    dataset = SRDataset(HR_DIR, LR_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SRCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    best_loss = float("inf")
    model_path = os.path.join(MODEL_DIR, "srcnn_trained_model.pth")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)

            preds = model(lr_imgs)
            loss = criterion(preds, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)

# Run
if __name__ == "__main__":
    train_model(epochs=20, batch_size=16, lr_rate=1e-4, resize_dim=128)
