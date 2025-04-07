import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

# ------------------ Config ------------------
dataset_path = '/home/sascha/kubernetes-intrusion-detection-main/KubeFocus/video/image_dataset.pt'
batch_size = 32
epochs = 10
learning_rate = 0.001
checkpoint_path = '/home/sascha/kubernetes-intrusion-detection-main/KubeFocus/artifacts/spatiotemporal_cnn.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Dataset ------------------
class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data  # List of {'image': ..., 'label': ...}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image'].permute(2, 0, 1).float() / 255.0  # [3, 32, 32], normalized
        label = item['label']
        return img, label

# ------------------ Load and Split ------------------
data = torch.load(dataset_path)
dataset = ImageDataset(data)

total_len = len(dataset)
train_len = int(0.7 * total_len)
val_len = int(0.15 * total_len)
test_len = total_len - train_len - val_len

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ------------------ Spatiotemporal CNN ------------------
class SpatiotemporalCNN(nn.Module):
    def __init__(self):
        super(SpatiotemporalCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2),    
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.conv(x).view(x.size(0), -1)

# ------------------ Training Setup ------------------
model = SpatiotemporalCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred

# ------------------ Training Loop ------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} loss: {running_loss / len(train_loader):.4f}")

    y_val_true, y_val_pred = evaluate(model, val_loader)
    print(f"Validation Report:\n{classification_report(y_val_true, y_val_pred, digits=4)}")

# ------------------ Final Test ------------------
y_test_true, y_test_pred = evaluate(model, test_loader)
print("🧪 Final Test Results:")
print(classification_report(y_test_true, y_test_pred, digits=4))

# ------------------ Save Checkpoint ------------------
torch.save(model.state_dict(), checkpoint_path)
print(f"✅ Model saved to {checkpoint_path}")
