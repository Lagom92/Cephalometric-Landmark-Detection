import os
import cv2
import torch
import shutil
from tqdm import tqdm

from config import *
from utils.func import *
from utils.dataset import CoarseDataset
from models.resnet50_dilated import ResNet_dilated as coarse_model


# Load model
print("coarse model path: ", coarse_model_path)
model_coarse = coarse_model()
if torch.cuda.is_available():
    model_coarse.load_state_dict(torch.load(coarse_model_path))
else:
    model_coarse.load_state_dict(torch.load(coarse_model_path, map_location=torch.device('cpu')))
model_coarse.to(device)
model_coarse.eval()
print("Load weight to model")

# Make rois dir
if os.path.exists(roi_path):
    shutil.rmtree(roi_path)
os.makedirs(roi_path)

for i in range(1, landmark_num_len+1):
    os.makedirs(f"{roi_path}{str(i)}")
print("Make rois dir")
    
print("Start create ROIs")
roi_cnt, error_cnt = 0, 0      # Success and error rois counts
with torch.no_grad():
    train_val_dataset = CoarseDataset(train_img_path, flag=True)
    for idx, val in enumerate(tqdm(train_val_dataset)):
        title, resized_img_, resized_labels, [org_img, org_labels] = val
        #resized_img_, resized_labels, [org_img, org_labels] = val
        #title = org_labels[0].split('.')[0]
        #org_labels = org_labels[1:]
        resized_img = resized_img_.unsqueeze(0).unsqueeze(0).to(device)

        # infer
        pred_labels = model_coarse(resized_img)
        pred_labels = pred_labels.cpu()

         # infer 결과를 원본 사이즈로 변환        
        changed_pred_labels = []
        for j in range(len(resized_labels)):
            change_ratio = resized_labels[j] / org_labels[j]
            val = pred_labels[0][j] / change_ratio
            changed_pred_labels.append(val)
        pred_labels = list(map(float, changed_pred_labels))

        # Class별 ROI and coordinates
        for k in range(0, len(pred_labels), 2):
            org_x = int(org_labels[k])
            org_y = int(org_labels[k+1])
            pred_x = int(pred_labels[k])
            pred_y = int(pred_labels[k+1])

            box = [0, 0, 0, 0]
            y1 = pred_y - img_size
            y2 = y1 + 2*img_size
            x1 = pred_x - img_size 
            x2 = x1 + 2*img_size

            lst = [y1, y2, x1, x2]

            for i, v in enumerate(lst):
                if v <= 0:
                    box[i] = -1*v
                    lst[i] = 0
                if i == 1:
                    num = v - org_img.shape[0]
                    if num > 0:
                        box[i] = num
                        lst[i] = org_img.shape[0]
                elif i == 3:
                    num = v - org_img.shape[1]
                    if num > 0:
                        box[i] = num
                        lst[i] = org_img.shape[1]
            [y1, y2, x1, x2] = lst

            img_cropped = org_img[y1:y2, x1:x2]

            gt_x = org_x - (pred_x - img_size)
            gt_y = org_y - (pred_y - img_size)

            # Padding 적용(448x448 size)
            img_cropped = cv2.copyMakeBorder(img_cropped, box[0], box[1], box[2], box[3], cv2.BORDER_CONSTANT)

            # Exception
            if img_cropped.shape != (img_size*2, img_size*2) or gt_x < 0 or gt_y < 0 or gt_x > img_size*2 or gt_y > img_size*2:
                error_cnt += 1
                break
            
            # Save ROI
            img_path = f"{roi_path}{(k//2)+1}/{title}.jpg"
            # cv2.line(img_cropped, (gt_x, gt_y), (gt_x, gt_y), (0, 0, 255), 10)
            cv2.imwrite(img_path, img_cropped)

            # Save Landmarks
            csv_path = f"{roi_path}{(k//2)+1}_landmark.csv"
            data = [f"{title}.jpg", gt_x, gt_y]
            mode = 'w' if not os.path.exists(csv_path) else 'a'
            write_coor(csv_path, data, mode)

            roi_cnt += 1

print(f"Total ROIs cnt: {roi_cnt}, Error cnt: {error_cnt}")
print("END")
