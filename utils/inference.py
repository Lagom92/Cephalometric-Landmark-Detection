import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

from config import *
from models.resnet50_dilated import ResNet_dilated as coarse_model
from models.resnet50 import ResNet50_fine as fine_model


# Preprocessing 1
''' 
전처리(CLAHE, Resize) 
output: preprocessed image
'''
def preprocessing_1(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(img)
    
    transform = A.Compose([
        A.Resize(224, 224),
        ToTensor()
    ])
    data = transform(image=image)
    
    return data['image']


# Network 1
''' 
학습된 1차 모델을 이용하여  coarse landmarks 감지
output: 38개의 좌표값(x1, y2, ... x19, y19)
'''
def network_1(resized_img, model_coarse):

    with torch.no_grad():
        res = model_coarse(resized_img.unsqueeze(0).unsqueeze(0))
    
    pred_landmarks = res[0].tolist()
    pred_landmarks = [round(pred, 4) for pred in pred_landmarks]
    
    return pred_landmarks


# Preprocessing 2
'''
1차 모델에서 구한 좌표(x, y)를 중심으로 하는 ROIs 생성 및 전처리
output: ROI 리스트, 원본이미지에서의 roi 시작 좌표
'''
def preprocessing_2(org_img, coarse_landmarks):
    img_size = 224
    rois = []    # ROI 이미지 모음
    rois_crop_coor = []     # Crop 위치

    # infer 결과를 원본 사이즈로 변환        
    h_ratio, w_ratio = np.array(org_img.shape) / img_size
    pred_landmarks = []
    for j in range(0, len(coarse_landmarks), 2):
        w_val = coarse_landmarks[j] * w_ratio
        pred_landmarks.append(w_val)
        h_val = coarse_landmarks[j+1] * h_ratio
        pred_landmarks.append(h_val)

    # Class별 ROIs
    for k in range(0, len(pred_landmarks), 2):
        box = [0, 0, 0, 0]
        pred_x = int(pred_landmarks[k])
        pred_y = int(pred_landmarks[k+1])

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

        rois_crop_coor.append(pred_y-img_size)
        rois_crop_coor.append(pred_x-img_size)

        # zero-padding
        img_cropped = cv2.copyMakeBorder(img_cropped, box[0], box[1], box[2], box[3], cv2.BORDER_CONSTANT)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_cropped = clahe.apply(img_cropped)
        
        # RESIZE
        transform = A.Compose([
            A.Resize(img_size, img_size),
            ToTensor()
        ])
        data = transform(image=image_cropped)

        img_roi = data['image']

        rois.append(img_roi)
    return rois, rois_crop_coor


# Network 2
'''
output: landmarks 좌표 리스트
'''
def network_2(img_roi, model_fine):
    with torch.no_grad():
        pred = model_fine(img_roi.unsqueeze(0).unsqueeze(0)).detach().cpu()
        pred_landmarks = list(v.item() for v in pred[0])

    return pred_landmarks


# Inference
def infer(image_path):
    org_img = cv2.imread(image_path, 0)

    pp_img = preprocessing_1(org_img)

    model_coarse = coarse_model()
    if torch.cuda.is_available():
        model_coarse.load_state_dict(torch.load(coarse_model_path))    # in GPU
    else:
        model_coarse.load_state_dict(torch.load(coarse_model_path, map_location=torch.device('cpu')))    # in CPU
    model_coarse.to(device)
    model_coarse.eval()

    coarse_landmarks = network_1(pp_img.to(device), model_coarse)

    rois, rois_crop_coor = preprocessing_2(org_img, coarse_landmarks)

    pred_lst = []
    for i, img_roi in enumerate(rois):
        cls_num = str(i+1)

        fine_model_path = os.path.join(fine_weight_path, f"{cls_num}_model.pt")
        
        model_fine = fine_model()
        model_fine.to(device)
        if torch.cuda.is_available():
            model_fine.load_state_dict(torch.load(fine_model_path))    # in GPU
        else:
            model_fine.load_state_dict(torch.load(fine_model_path, map_location=torch.device('cpu')))    # in CPU
        model_fine.eval()
        
        pred = network_2(img_roi.to(device), model_fine)
        pred_lst.extend(pred)
    
    pred_lst_224 = list(v*2 for v in pred_lst)

    pred_landmarks = []
    for i in range(0, len(pred_lst_224), 2):
        y, x = pred_lst_224[i+1] + rois_crop_coor[i], pred_lst_224[i] + rois_crop_coor[i+1]
        pred_landmarks.append(int(x))
        pred_landmarks.append(int(y))

    return pred_landmarks

