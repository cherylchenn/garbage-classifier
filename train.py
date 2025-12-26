import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from datasets import load_dataset

# config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device:", DEVICE)
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-4
VAL_SPLIT = 0.2 # 80% train vs 20% val

# dataset wrapper
class TrashDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataset():
    ds = load_dataset("garythung/trashnet")["train"]
    classes = ds.features["label"].names

    # split into train / validation
    split = ds.train_test_split(test_size=VAL_SPLIT, seed=42)
    train_ds = split["train"]
    test_ds = split["test"]
    return train_ds, test_ds, classes

def get_dataloaders(train_ds, test_ds, transform):
    train_loader = DataLoader(TrashDataset(train_ds, transform), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(TrashDataset(test_ds, transform),
                             batch_size=BATCH_SIZE, num_workers=4)
    return train_loader, test_loader

def build_model(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

def save_labels(classes, filename="labels.txt"):
    with open(filename, "w") as f:
        for c in classes:
            f.write(c + "\n")

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(loader)

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def train_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        avg_loss = train(model, train_loader, criterion, optimizer)
        val_acc = validate(model, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}%")
    
    torch.save(model.state_dict(), "model.pth")
    print("Training completed! Model saved as model.pth!")

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], 
                             std=[0.229,0.224,0.225])
    ])

    train_ds, test_ds, classes = get_dataset()
    print("Dataset loaded!")
    save_labels(classes)
    print(f"Classes:", classes)
    train_loader, test_loader = get_dataloaders(train_ds, test_ds, transform)
    print("Building model...")
    model = build_model(len(classes))
    print("Model built! Starting training...")
    train_model(model, train_loader, test_loader)