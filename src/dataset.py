import numpy as np
import torch as th
from torch.utils.data import Dataset
from torch import LongTensor

class CustomDataset(Dataset):
    def __init__(self, data_split, prefetch_gpu=False):
        self.prefetch_gpu = prefetch_gpu
        self.dataset = np.ascontiguousarray(data_split)

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]
    def shuffle(self):
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()
        self.dataset = LongTensor(np.ascontiguousarray(np.random.shuffle(self.dataset)))
        if self.prefect_gpu:
            self.dataset = self.dataset.cuda().contiguous()
