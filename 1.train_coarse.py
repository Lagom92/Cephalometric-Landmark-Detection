import os
import csv
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from config import *
from utils.func import *
from utils.dataset import CoarseDataset
from models.resnet50_dilated import ResNet_dilated as coarse_model


# Dataset
train_val_dataset = CoarseDataset(train_img_path, flag=False)
test_dataset = CoarseDataset(test_img_path, flag=False)

len_val_set = int(0.1 * len(train_val_dataset))
len_train_set = len(train_val_dataset) - len_val_set

train_dataset, val_dataset = random_split(train_val_dataset, [len_train_set, len_val_set], generator=torch.Generator().manual_seed(seed))

print(f"train : val : test = {len(train_dataset)} : {len(val_dataset)} : {len(test_dataset)}")

# Dataloader
train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True)
test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True)

# Make dir
if os.path.exists(weight_path):
    shutil.rmtree(weight_path)
os.makedirs(weight_path)
f = open(f"{weight_path}coarse_log.csv", 'w', newline='')
wr = csv.writer(f)
wr.writerow(['epoch', 'train loss', 'val loss', 'LR'])
f.close()

''' Training Coarse Model '''
# Make network
model = coarse_model()
model.to(device)
print("Create coarse model")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=5, after_scheduler=cosine_scheduler)

early_stopping = EarlyStopping(patience=20, path=coarse_model_path, verbose=True)

# Training
print("Start Training...")
save_point = 0
min_loss = float('inf')

for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0
    
    model.train()
    for batch_index, (title, features, labels) in enumerate(train_data):
        img = features.to(device).unsqueeze(1)
        labels = torch.stack(labels).view(batch_size, -1)
        labels = np.array(labels, dtype=np.float32)
        labels = torch.as_tensor(labels).to(device)
        
        optimizer.zero_grad()

        outputs = model(img)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += (loss.item() / batch_size)
    
    mean_train_loss = train_loss / len(train_data)

    model.eval()
    with torch.no_grad():
        for batch_index, (title, features, labels) in enumerate(val_data):
            img = features.to(device).unsqueeze(1)
            labels = torch.stack(labels).view(batch_size, -1)
            labels = np.array(labels, dtype=np.float32)
            labels = torch.as_tensor(labels).to(device)

            outputs = model(img)

            loss = criterion(outputs, labels)
            
            val_loss += (loss.item() / batch_size)
            
    mean_val_loss = val_loss / len(val_data)
    
    lr_scheduler.step()
    early_stopping(mean_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
    # print log
    print(f"epoch: {epoch+1} | training loss: {mean_train_loss:.8f} | validation loss: {mean_val_loss:.8f} | lr: {optimizer.param_groups[0]['lr']:.10f}")
    
    # Save log
    f = open(f"{weight_path}coarse_log.csv", 'a', newline='')
    wr = csv.writer(f)
    wr.writerow([epoch+1, train_loss / len(train_data), mean_val_loss, optimizer.param_groups[0]['lr']])
    f.close()  

    if mean_val_loss < min_loss:
        min_loss = mean_val_loss
        save_point = epoch+1
        torch.save(model.state_dict(), coarse_model_path)
        
print(f"Min val loss: {min_loss} | Save idx: {save_point}")
print("End Training")
