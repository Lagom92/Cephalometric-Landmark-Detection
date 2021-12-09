import os
import pandas as pd
import shutil
import cv2
from tqdm import tqdm

from config import *
from utils.func import *
from utils.inference import infer


# Make result dir
print("Result path: ", result_path)
if os.path.exists(result_path):
    shutil.rmtree(result_path)
os.makedirs(result_path)

# Create result csv
csv_path = result_path + 'pred_annotation.csv'
header_lst = ['image_name','x1','y1','x2','y2','x3','y3','x4','y4','x5','y5', 'x6','y6','x7','y7','x8','y8','x9','y9','x10','y10','x11','y11','x12','y12',
            'x13','y13','x14','y14','x15','y15','x16','y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19']
write_coor(csv_path, data=header_lst, mode='w')


test_lst = os.listdir(test_img_path)
img_lst = []
for data in test_lst:
    if data[-4:] == 'jpeg':
        img_lst.append(data)
n = len(img_lst)

table = {}
for idx in tqdm(range(n)):
    json_path = test_img_path + img_lst[idx][:-4] + 'json'
    img_path = test_img_path + img_lst[idx]
    org_landmarks = json2lst(json_path)

    # Inference
    pred_landmarks = infer(img_path)

    # Convert float to int
    pred_landmarks = list(map(int, pred_landmarks))
    org_landmarks = list(map(int, org_landmarks))

    dcm_path = test_img_path + img_lst[idx][:-4] +'dcm' 
    row, col = get_pixelspacing(dcm_path)
    distance = cal_distance(org_landmarks, pred_landmarks, row, col)
    for m in range(0, len(distance)):
        num = str(m+1)
        val = round(distance[m], 4)
        if num in table.keys():
            table[num].append(val)
        else:
            table[num] = [val]
    
    # Save predict result
    data = [img_lst[idx]] + pred_landmarks
    write_coor(csv_path, data=data, mode='a')
    
    # Save Image with landmarks
    img_rgb = cv2.imread(img_path)
    for k in range(0, len(pred_landmarks), 2):
        gt_x, gt_y = org_landmarks[k], org_landmarks[k+1]
        pred_x, pred_y = pred_landmarks[k], pred_landmarks[k+1]

        cv2.line(img_rgb, (pred_x, pred_y), (pred_x, pred_y), (255, 0, 0), 20)
        cv2.putText(img_rgb, str((k//2)+1), (pred_x+10, pred_y-10), fontFace=0, fontScale=1, color=(255, 0, 0), thickness=3)
        cv2.line(img_rgb, (gt_x, gt_y), (gt_x, gt_y), (0, 0, 255), 20)
        cv2.putText(img_rgb, str((k//2)+1), (gt_x+10, gt_y-10), fontFace=0, fontScale=1, color=(0, 0, 255), thickness=3)
    cv2.imwrite(f"{result_path}{img_lst[idx]}", img_rgb)

# Show results
print(f"\ntest dataset counts: {len(img_lst)}")
print("Average distance by class (mm)")
n = len(table)
total_avg = 0
for idx, val in table.items():
    avg = sum(val)/len(val)
    print(f"landmark: {idx} | {round(avg, 2)}")  
    total_avg += avg
total_avg /= n
print("Total avg: ", round(total_avg, 2))
print("END")
