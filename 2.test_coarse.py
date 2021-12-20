import os
import cv2
import torch
import shutil
from tqdm import tqdm

from config import *
from utils.func import *
from utils.dataset import CoarseDataset
from models.resnet50_dilated import ResNet_dilated as coarse_model


test_dataset = CoarseDataset(test_img_path, flag=True)
print(f"len test: {len(test_dataset)}")

# Make dir
if os.path.exists(labeled_img_path):
    shutil.rmtree(labeled_img_path)
os.makedirs(labeled_img_path)

# Save Coarse test landmark
pred_csv_path = f"{labeled_img_path}pred_landmarks.csv"
data = ['image_name','x1','y1','x2','y2','x3','y3','x4','y4','x5','y5','x6','y6','x7','y7','x8','y8','x9','y9','x10','y10','x11','y11','x12','y12',
        'x13','y13','x14','y14','x15','y15','x16','y16','x17','y17','x18','y18','x19','y19']
write_coor(pred_csv_path, data, 'w')

# Save origin test landmark
org_csv_path = f"{labeled_img_path}org_landmarks.csv"
write_coor(org_csv_path, data, 'w')

# Load model
model = coarse_model()
if torch.cuda.is_available():
    model.load_state_dict(torch.load(coarse_model_path))
else:
    model.load_state_dict(torch.load(coarse_model_path, map_location=torch.device('cpu')))
model.to(device)
model.eval()
print("Load weight to model")

table = {}
with torch.no_grad():
    for idx, (title, resized_img, resized_labels, [org_img, org_labels]) in enumerate(tqdm(test_dataset)):
        resized_img = resized_img.unsqueeze(0).unsqueeze(0).to(device)
        
        # Infer
        pred_labels = model(resized_img)
        pred_labels = pred_labels.cpu()
        
         # infer 결과를 원본 사이즈로 변환
        changed_pred_labels = []
        for j in range(len(resized_labels)):
            change_ratio = torch.true_divide(resized_labels[j], org_labels[j])
            val = pred_labels[0][j] / change_ratio
            changed_pred_labels.append(val)

        # Convert type(float to int)
        org_labels = list(map(int, org_labels))
        pred_labels = list(map(int, changed_pred_labels))
        
        # Save Coarse test images
        pred_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
        for idx in range(0, len(pred_labels), 2):
            gx, gy = org_labels[idx], org_labels[idx+1]
            px, py = pred_labels[idx], pred_labels[idx+1]
            cv2.line(pred_img, (gx, gy), (gx, gy), (0, 0, 255), 15)
            cv2.line(pred_img, (px, py), (px, py), (255, 0, 0), 15)
        cv2.imwrite(f"{labeled_img_path}{title}.jpg", pred_img)

        # Save Coarse test landmark
        data = [f"{title}.jpeg"] + pred_labels
        write_coor(pred_csv_path, data, 'a')

        # Save origin test landmark
        data = [f"{title}.jpeg"] + org_labels
        write_coor(org_csv_path, data, 'a')

        # Measure distance
        distance = cal_distance(org_labels, pred_labels)
        for m in range(0, len(distance)):
            num = str(m+1)
            val = round(distance[m], 4)
            if num in table.keys():
                table[num].append(val)
            else:
                table[num] = [val]
                           
# Show results
print(f"\ntest dataset counts: {len(test_dataset)}")
print("Average distance by class (pixel)")
n = len(table)
total_avg = 0
for idx, val in table.items():
    avg = sum(val)/len(val)
    print(f"landmark: {idx} | {round(avg, 2)}")  
    total_avg += avg
total_avg /= n
print("Total avg: ", round(total_avg, 2))
