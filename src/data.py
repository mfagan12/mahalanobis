import torch
import torchvision
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="../data/"):
        self.data_dir = data_dir

    def prepare_data(self):
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


class KMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="../data/"):
        self.data_dir = data_dir

    def prepare_data(self):
        torchvision.datasets.KMNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.KMNIST(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


if __name__ == "__main__":
    pass
