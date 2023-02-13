from .synthetic_datasets import *
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        x = self.X_data[index]
        y = self.y_data[index]
        return x, y

    def __len__(self):
        return len(self.X_data)
