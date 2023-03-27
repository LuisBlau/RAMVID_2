import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_data(*, data_dir, batch_size, image_size, num_workers, rgb, seq_len):
    data_file = data_dir  # Update the path to your Moving MNIST dataset
    batch_size = batch_size
    num_workers = num_workers
    dataset = MovingMNISTDataset(data_file=data_file)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True
    )

    while True:
        yield from loader


class MovingMNISTDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data_file = data_file
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = np.load(self.data_file)
        data = np.expand_dims(data, axis=1)  # Add channel dimension (C=1)
        data = np.transpose(data, [2,1,0,3,4])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)
        sample = sample.astype(np.float32)
        #print(sample.shape)
        return sample


