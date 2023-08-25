import numpy as np
import torch
from torch.utils.data import Dataset


class SummaryDataset(Dataset):
    def __init__(self, data, para, norm=True):

        if norm:
            self.total_data = self.preprocess(data)
            self.total_para = self.preprocess(para)
        else:
            self.total_data = data
            self.total_para = para

    def __len__(self):
        return (self.total_data.shape[0])

    @classmethod
    def preprocess(cls, data, mean=None, std=None):

        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    def __getitem__(self, i):
        data = self.total_data[i]
        param = self.total_para[i]

        return {
            'data': torch.from_numpy(data).type(torch.FloatTensor),
            'param': torch.from_numpy(param).type(torch.FloatTensor)
        }
