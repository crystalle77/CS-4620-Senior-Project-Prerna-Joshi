import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
import numpy as np

# === CONFIG ===
disease_labels = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding"
]
BATCH_SIZE = 64
EPOCHS = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "saved_models"  # Directory to save models

# Ensure the save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === DATASET CLASS ===
class CheXpertDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
        self.labels = disease_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['real_path']  # Path is already in the file, no need for matching.csv
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load image at {img_path}: {e}")
            raise
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[self.labels].values.astype(np.float32))
        return image, label

# === HELPER: Create Model ===
def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(disease_labels))
    return model.to(device)

# === HELPER: Train and Evaluate ===
def train_and_evaluate(train_df, test_dfs, model_name):
    train_dataset = CheXpertDataset(train_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n=== Training {model_name} Model ===")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {total_loss / len(train_loader):.4f}")

    # === Evaluation on multiple datasets ===
    model.eval()
    for test_df, test_name in test_dfs:
        test_dataset = CheXpertDataset(test_df, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                y_true.append(labels.cpu().numpy())
                y_pred.append(torch.sigmoid(outputs).cpu().numpy())

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)

        print(f"\n=== {model_name} AUC Scores on {test_name} ===")
        for i, label in enumerate(disease_labels):
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            except ValueError:
                auc = float('nan')
            print(f"{label}: {auc:.4f}")

    # === Save the model ===
    model_save_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# === LOAD AND CLEAN DATASETS ===
def load_and_prepare(path):
    df = pd.read_csv(path)
    df[disease_labels] = df[disease_labels].fillna(0).replace(-1, 0)
    return df

# Load datasets
train_single = load_and_prepare("train_single_race.csv")
test_same = load_and_prepare("test_same_race.csv")
test_diff = load_and_prepare("test_other_races.csv")
#train_mixed = load_and_prepare("train_mixed_race.csv")

# Ensure the column storing image paths is named correctly (e.g., "Path")
train_single = train_single[train_single['real_path'].apply(lambda x: isinstance(x, str) and x.strip() != '')].copy()
test_same = test_same[test_same['real_path'].apply(lambda x: isinstance(x, str) and x.strip() != '')].copy()
test_diff = test_diff[test_diff['real_path'].apply(lambda x: isinstance(x, str) and x.strip() != '')].copy()
#train_mixed = train_mixed[train_mixed['real_path'].apply(lambda x: isinstance(x, str) and x.strip() != '')].copy()


# === RUN TRAINING AND EVALUATION ON BOTH TEST SETS ===
test_dfs = [
    (test_same, "Same-Race"),
    (test_diff, "Different-Race")
]

train_and_evaluate(train_single, test_dfs, model_name="Same-Race")