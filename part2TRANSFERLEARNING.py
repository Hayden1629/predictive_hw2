
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import pandas as pd
import numpy as np
from keras import layers
from pathlib import Path
os.environ["KERAS_BACKEND"] = "torch"
keras.utils.set_random_seed(58008)
print("CUDA available:", torch.cuda.is_available())

IMG_SIZE = 224
BATCH_SIZE = 64 # adjust based on vram usage

train_paths = list(Path('./data/train').glob('*.jpg'))
train_df  = pd.DataFrame({
    'filepath': [str(p) for p in train_paths],
    'label': [1 if p.stem.startswith('dog') else 0 for p in train_paths]
})
train_df = train_df.sample(frac=1, random_state = 58008).reset_index(drop=True)

from keras.utils import image_dataset_from_directory

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]['filepath']).convert('RGB')
        label = self.df.iloc[idx]['label']
        if self.transform:
            img = self.transform(img)
        return img, label


import torchvision.transforms as T
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, .486, .406], [.229, .224, .225]) #these stats are from ImageNet
])

split = int(.8 * len(train_df))
train_ds = CatDogDataset(train_df[:split], transform=train_transform)
val_ds = CatDogDataset(train_df[split:], transform=T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, .486, .406], [.229, .224, .225])
]))

train_loader = DataLoader(train_ds, batch_size= BATCH_SIZE, shuffle = True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

import torchvision.models as models
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device', device)

model = models.efficientnet_b0(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(.3),
    nn.Linear(model.classifier[1].in_features, 1),
    nn.Sigmoid()
)
model = model.to(device)


import time

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=.004)
criteron = nn.BCELoss()

for epoch in range(5):
    print("starting epoch ", epoch)
    model.train()
    start = time.time()
    for i, (imgs, labels) in enumerate(train_loader):
        
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        preds = model(imgs).squeeze()
        loss = criteron(preds, labels)
        loss.backward()
        optimizer.step()

        if i ==10: #10 batches
            elapsed = time.time() - start
            est_epoch = elapsed * len(train_loader)
            print(f"~{est_epoch/60:.1f} min per epoch")
            break

    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            preds = model(imgs).squeeze()
            correct += ((preds > .5) == labels).sum().item()
            total += len(labels)
    print(f'Epoch {epoch+1} | Val Acc: {correct/total:.4f}')


test_paths = sorted(Path('./data/test').glob('*jpg'), key=lambda p: int(p.stem))

test_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, .486, .406], [.229, .224, .225])
])

ids, preds = [], []
model.eval()
with torch.no_grad():
    for path in test_paths:
        img = Image.open(path).convert("RGB")
        img = test_transform(img).unsqueeze(0).to(device)
        prob = model(img).item()
        ids.append(int(path.stem))
        preds.append(prob)

submission = pd.DataFrame({'id': ids, 'label':preds})
submission.to_csv('submission.csv', index=False)