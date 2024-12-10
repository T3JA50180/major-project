import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import lightning as L
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch

class APTOSDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.data.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


class DataModule(L.LightningDataModule):
    def __init__(self, train_csv, train_dir, batch_size, num_workers, pin_memory):
        super().__init__()
        self.train_csv = train_csv
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage):
        dataset = APTOSDataset(csv_file=self.train_csv, images_dir=self.train_dir, transform=self.transforms)
        self.train_dataset, self.val_dataset = random_split(
            dataset=dataset, lengths=[0.9, 0.1]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
