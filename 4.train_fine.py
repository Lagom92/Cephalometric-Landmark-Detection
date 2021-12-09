import os
import csv
import time
import shutil
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from config import *
from utils.func import *
from utils.dataset import FineDataset
from models.resnet50 import ResNet50_fine as fine_model


# Make fine model dir
if os.path.exists(fine_weight_path):
    shutil.rmtree(fine_weight_path)
os.makedirs(fine_weight_path)

total_save_point = []
total_train_loss_hist = []
total_val_loss_hist = []
min_loss_hist = []

for num in tqdm(landmark_num):
    print(f"------ Training classes: {num} ------")
    f = open(f"{fine_weight_path}{num}_log.csv", 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['epoch', 'train loss', 'val loss', 'LR'])
    f.close()
    
    # Dataset
    dataset = FineDataset(num, f"{roi_path}{num}_landmark.csv")
    
    # Split train and val
    n = len(dataset)
    len_train_set = int(0.9 * n)
    len_val_set = n - len_train_set

    train_dataset, val_dataset = random_split(dataset, [len_train_set, len_val_set], generator=torch.Generator().manual_seed(seed))
    
    print("totla : ", n)
    print(f"train : val = {len(train_dataset)} : {len(val_dataset)}")

    # Dataloader
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    
    # Make fine model
    model = fine_model()
    model.to(device)
    print("Create fine model")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate2, weight_decay=weight_decay2)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=5, after_scheduler=cosine_scheduler)
    
    early_stopping = EarlyStopping(patience=20, path=f'{fine_weight_path}{num}_model.pt', verbose=False)
    
    optimizer.zero_grad()
    
    save_point = 0
    min_loss = float('inf')

    for epoch in range(epochs2):
        start_time = time.time()
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for batch_index, (features, labels, org) in enumerate(train_data):
            img = features.to(device).unsqueeze(1)
            labels = torch.stack(labels).view(batch_size, -1)
            labels = np.array(labels, dtype=np.float32)
            labels = torch.as_tensor(labels).to(device)
            
            outputs = model(img)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        mean_train_loss = train_loss / len(train_data)

        model.eval()
        with torch.no_grad():
            for batch_index, (features, labels, org) in enumerate(val_data):
                img = features.to(device).unsqueeze(1)
                labels = torch.stack(labels).view(batch_size, -1)
                labels = np.array(labels, dtype=np.float32)
                labels = torch.as_tensor(labels).to(device)

                outputs = model(img)
                
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        mean_val_loss = val_loss / len(val_data)

        lr_scheduler.step()
        
        early_stopping(mean_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f"epoch: {epoch+1} | training loss: {mean_train_loss:.8f} | validation loss: {mean_val_loss:.8f} | lr: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Save log
        f = open(f"{fine_weight_path}{num}_log.csv", 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([epoch+1, mean_train_loss, mean_val_loss, optimizer.param_groups[0]['lr']])
        f.close()          
        
        if mean_val_loss < min_loss:
            min_loss = mean_val_loss
            save_point = epoch+1
            torch.save(model.state_dict(), f'{fine_weight_path}{num}_model.pt')
    
        elapsed_time = time.time() - start_time

        print("Spend time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print(f"Landmark: {num} | min val loss: {min_loss} | Save idx: {save_point}")
    
print("End Training")
