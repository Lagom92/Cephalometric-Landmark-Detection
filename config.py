import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from utils.func import random_seed

# Ignore warning
import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

'''Path 설정'''
base_path = 'base/'

train_img_path = './data/train/'
test_img_path = './data/test/'

weight_path = f"./weights/{base_path}"
coarse_model_path = f"{weight_path}coarse_model.pt"
fine_weight_path = f"{weight_path}fine/"

labeled_img_path = './result/coarse/'

roi_path = './data/roi/'

result_path = f"./result/{base_path}"

img_size = 224
seed = 0
random_seed(seed)

# 랜드마크 번호 및 개수
# ['S', 'Po', 'N', 'Or', 'A', "Pog'", 'B', 'PNS', 'ANS', 'Ar', 'Me', 'U1', 'L1', 'Pn', 'Ls', 'Li', 'Pog', 'Go', 'Gn']
landmark_num = ['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
landmark_num_len = len(landmark_num)

'''Parameters'''
batch_size = 1
workers = 4
T_max = 50

# 1차 네트워크
epochs = 100
learning_rate = 1e-7

# 2차 네트워크
epochs2 = 10
learning_rate2 = 1e-6
weight_decay2 = 1e-5
