import torch
import random
import os 
import numpy as np
from scipy.linalg import fractional_matrix_power
from torch.utils.data import Subset
def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def EA(x, ref=False):

    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1])) #(bs,channel,channel)
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(x.shape[1])
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    if ref:
        return XEA, sqrtRefEA
    else:
        return XEA
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # Check if dataset is a Subset and access original dataset
        if isinstance(dataset1, Subset):
            self.dataset1 = dataset1.dataset
            self.indices1 = dataset1.indices
        else:
            self.indices1 = list(range(len(dataset1)))

        if isinstance(dataset2, Subset):
            self.dataset2 = dataset2.dataset
            self.indices2 = dataset2.indices
        else:
            self.indices2 = list(range(len(dataset2)))

        # Combine data and labels
        self.data = np.array([self.dataset1[i][0] for i in self.indices1] +
                             [self.dataset2[i][0] for i in self.indices2])

        self.labels = np.array([self.dataset1[i][1] if torch.is_tensor(self.dataset1[i][1])
                                else self.dataset1[i][1] for i in self.indices1] +
                               [self.dataset2[i][1] if torch.is_tensor(self.dataset2[i][1])
                                else self.dataset2[i][1] for i in self.indices2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
class EarlyStopping:
    def __init__(self, patience=10, path=None):
        self.patience = patience
        self.counter = 0
        self.val_min_acc = 0.
        self.early_stop = False
        self.path = path

    def __call__(self, val_acc, model):
        if val_acc < self.val_min_acc:
            self.counter += 1
            if self.counter >= self.patience:
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.val_min_acc = val_acc
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)