import numpy as np
from torch.utils.data import Dataset


class DefaultDataset(Dataset):
    def __init__(self, X: np.array, y: np.array, transform=None):

        self.X = X.astype("float32")
        self.y = y.astype("int32")

        self.n_samples = self.X.shape[0]
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        sample = self.X[idx], self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
