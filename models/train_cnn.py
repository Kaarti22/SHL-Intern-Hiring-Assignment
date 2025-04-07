import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from models.cnn_model import AudioCNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx]["filename"].replace(".wav", ".png")
        label = self.data.iloc[idx]["label"]
        image = Image.open(os.path.join(self.image_folder, filename)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = AudioDataset("data/train.csv", "data/mel_spectrograms_train", transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    running_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "outputs/model_cnn.pth")
print("Model saved to outputs/model_cnn.pth")

model.eval()
with torch.no_grad():
    y_preds = []
    y_trues = []
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        y_preds.extend(outputs.cpu().numpy().flatten())
        y_trues.extend(labels.numpy().flatten())

y_preds = np.array(y_preds)
y_trues = np.array(y_trues)

mse = mean_squared_error(y_trues, y_preds)
pearson_corr, _ = pearsonr(y_trues, y_preds)
print(f"Validation MSE: {mse:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_trues, y=y_preds, edgecolor="w")
plt.xlabel("Actual Grammar Score")
plt.ylabel("Predicted Grammar Score")
plt.title("Actual vs Predicted Scores (CNN)")
plt.grid(True)
plt.savefig("outputs/cnn_actual_vs_predicted.png")
plt.show()

residuals = y_trues - y_preds
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, bins=20, color='skyblue')
plt.title("Residuals Distribution (Actual - Predicted)")
plt.xlabel("Error")
plt.ylabel("Count")
plt.grid(True)
plt.savefig("outputs/cnn_residual_distribution.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(y_preds, kde=True, bins=20, color='lightgreen')
plt.title("Distribution of Predicted Grammar Scores")
plt.xlabel("Predicted Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("outputs/cnn_predicted_score_distribution.png")
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_trues, y_preds, alpha=0.7)
z = np.polyfit(y_trues, y_preds, 1)
p = np.poly1d(z)
plt.plot(y_trues, p(y_trues), color="red", linestyle="--", label="Best fit line")
plt.xlabel("Actual Grammar Score")
plt.ylabel("Predicted Grammar Score")
plt.title("Actual vs Predicted with Best Fit Line (CNN)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/cnn_actual_vs_predicted_with_line.png")
plt.show()