import os
import cv2
import pandas as pd
from glob import glob
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

from config import *
from utils.func import json2lst

'''
path: input image 폴더
flag: True(original 데이터 출력 O) / False(original 데이터 출력 X)
'''
class CoarseDataset(Dataset):
    def __init__(self, path, flag=False):
        self.img_dir = path
        self.image_lst = glob(self.img_dir + '*.jpeg')
        self.img_size = img_size
        self.mode = flag
                
    def __len__(self):
        return len(self.image_lst)
    
    def __getitem__(self, index):
        img_path = self.image_lst[index]
        img_title = img_path.split(f"{self.mode}/")[-1]
        title = ''.join(img_title.split('.jpeg')[0])
        json_path = f"{self.img_dir}{title}.json"
        landmark = json2lst(json_path)
                                
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(img)

        org_size = image.shape
        w_ratio, h_ratio = org_size[0]/self.img_size, org_size[1]/self.img_size
        transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensor()
        ])
        data = transform(image=image)
        res = []
        for idx in range(0, len(landmark), 2):
                res.append(int(landmark[idx]/h_ratio))
                res.append(int(landmark[idx+1]/w_ratio))

        if self.mode:
            return title, data['image'], res, [img, landmark]
        else:
            return title, data['image'], res


class FineDataset(Dataset):
    def __init__(self, num, anno_path):
        self.root_dir = roi_path
        self.img_dir = self.root_dir + num + '/'
        self.image_lst = os.listdir(self.img_dir)
        self.landmarks = []
        self.img_size = img_size
        
        df = pd.read_csv(anno_path, header=None)  
        
        for i in range(df.shape[0]):
            sr = df.iloc[i].tolist()
            self.landmarks.append(sr)
                
    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, index):
        landmark = self.landmarks[index]
                        
        img_path = ''
        for path in self.image_lst:            
            if path == landmark[0]:
                img_path = self.img_dir + path
                
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(img)
        
        org_size = image.shape
        w_ratio, h_ratio = org_size[0]/self.img_size, org_size[1]/self.img_size
        transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensor()
        ])
        data = transform(image=image)
        res = []
        for idx in range(1, len(landmark), 2):
            res.append(int(landmark[idx]/h_ratio))
            res.append(int(landmark[idx+1]/w_ratio))

        return data['image'], res, [img, landmark]