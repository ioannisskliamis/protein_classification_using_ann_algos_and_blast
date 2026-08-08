from torch.utils.data import Dataset, DataLoader
import torch


class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx], dtype = torch.float32)
        x = x.view(x.size(0), -1)
        #x = x.unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype = torch.long)

        return x, y


class TestDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx], dtype = torch.float32)
        x = x.view(x.size(0), -1)
        #x = x.unsqueeze(0)

        return x