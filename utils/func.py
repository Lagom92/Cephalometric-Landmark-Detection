import torch
import numpy as np
import csv
import math
import pydicom
import torch
import json
from torch.optim.lr_scheduler import _LRScheduler


def random_seed(seed=0):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    return True

def get_pixelspacing(dcm_path):
    dcm_data = pydicom.dcmread(dcm_path)
    row, col = dcm_data.get('PixelSpacing', 'None')
    return row, col

def cal_distance(org_label, pred_label, row, col):
    res = []
    for k in range(0, len(org_label), 2):
        d1 = (org_label[k] - pred_label[k]) * row
        d2 = (org_label[k+1] - pred_label[k+1]) * col
        d = math.sqrt(d1**2 + d2**2)
        res.append(d)
    return res

def write_coor(csv_path, data, mode):
    with open(csv_path, mode, newline='') as f:
        w = csv.writer(f)
        w.writerow(data)
    return True

def json2lst(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        lst = []
        key_lst = list(data["annotation"].keys())
        for k in key_lst:
            coordinate = data['annotation'][k]['coordinate']
            lst.extend(coordinate)
    return lst

# Reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


'''
ex)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=5, after_scheduler=cosine_scheduler)
'''
# Reference: https://github.com/seominseok0429/pytorch-warmup-cosine-lr/blob/master/warmup_scheduler/scheduler.py
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)