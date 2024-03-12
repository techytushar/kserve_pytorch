import logging

import hydra
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST

from transforms import image_transforms


def download_mnist_data(train_only: bool = False) -> tuple[MNIST, MNIST | None]:
    train = MNIST(
        "../data",
        train=True,
        download=True,
        transform=image_transforms,
    )
    logging.info("Train data size: %s", len(train))

    if train_only:
        return train, None

    test = MNIST(
        "../data",
        train=False,
        download=True,
        transform=image_transforms,
    )
    logging.info("Test data size: %s", len(test))

    return train, test


@hydra.main(config_path="conf", config_name="config")
def get_data_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader | None]:
    train_data, test_data = download_mnist_data()

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )
    logging.info("Train loader created")

    if not test_data:
        return train_loader, None

    test_loader = DataLoader(
        test_data,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )
    logging.info("Test loader created")
    return train_loader, test_loader
