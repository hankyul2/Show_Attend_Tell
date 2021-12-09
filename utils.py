import math
import random
from copy import deepcopy

from pytorch_lightning import Callback
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet

import numpy as np
from rich.table import Table
from rich.console import Console
from timm.data import create_transform


class MyImageNet2012(ImageNet):
    def __init__(self, *args, download=False, train=True, **kwargs):
        kwargs['root'] = '/home/hankyul/hdd_ext/imageNet'
        kwargs['split'] = 'train' if train else 'val'
        super(MyImageNet2012, self).__init__(*args, **kwargs)


class BaseDataModule(LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 size: tuple,
                 augmentation: str = 'normal',
                 batch_size: int = 64,
                 num_workers: int = 4,
                 data_root: str = 'data',
                 valid_ratio: float = 0.1):
        """
        Base Data Module
        :arg
            Dataset: Enter Dataset
            batch_size: Enter batch size
            num_workers: Enter number of workers
            size: Enter resized image
            data_root: Enter root data folder name
            valid_ratio: Enter valid dataset ratio
        """
        super(BaseDataModule, self).__init__()

        if dataset_name == 'cifar10':
            dataset, self.mean, self.std = CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif dataset_name == 'cifar100':
            dataset, self.mean, self.std = CIFAR100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        elif dataset_name == 'imagenet':
            dataset, self.mean, self.std = MyImageNet2012, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        self.size = self.eval_size = size
        self.train_transform, self.test_transform = self.get_transforms(augmentation)
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.valid_ratio = valid_ratio
        self.num_classes = None
        self.num_step = None
        self.train_data_len = None
        self.test_data_len = None
        self.prepare_data()

    def get_transforms(self, augmentation):
        if augmentation == 'normal':
            train = transforms.Compose([
                transforms.Resize(self.size),
                transforms.Pad(4, padding_mode='reflect'),
                transforms.RandomCrop(self.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        elif augmentation.startswith('rand'):
            train = create_transform(self.size, is_training=True, auto_augment=augmentation, mean=self.mean, std=self.std)
        test = transforms.Compose([
            transforms.Resize(self.eval_size),
            transforms.CenterCrop(self.eval_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        return train, test

    def prepare_data(self) -> None:
        train = self.dataset(root=self.data_root, train=True, download=True)
        test = self.dataset(root=self.data_root, train=False, download=True)

        self.train_data_len = len(train)
        self.test_data_len = len(test)
        self.num_step = int(math.ceil(len(train) / self.batch_size))
        self.num_classes = len(train.classes)

    def setup(self, stage: str = None):
        self.train_ds = self.dataset(root=self.data_root, train=True, transform=self.train_transform)
        self.valid_ds = self.dataset(root=self.data_root, train=False, transform=self.test_transform)
        self.test_ds = self.dataset(root=self.data_root, train=False, transform=self.test_transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # Todo: Setting persistent worker makes server very slow!
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def make_random_mask(H, W):
    ratio = np.random.beta(1.0, 1.0)
    h, w = int(math.sqrt(1 - ratio) * H), int(math.sqrt(1 - ratio) * W)
    row, col = random.randint(0, H - h), random.randint(0, W - w)
    mask = torch.ones((H, W))
    mask[row:row + h, col:col + w] = 0
    ratio = 1 - (h * w) / (H * W)
    return mask, ratio


def cutmix(x, y):
    B, C, H, W = x.shape
    mask, ratio = make_random_mask(H, W)
    mask, rand_idx = mask.to(x.device), torch.randperm(B).to(x.device)
    return mask * x + (1 - mask) * x[rand_idx], y, y[rand_idx], ratio


def cutout(x, y):
    B, C, H, W = x.shape
    mask, ratio = make_random_mask(H, W)
    return mask.to(x.device) * x, y, ratio


def mixup(x, y):
    ratio = np.random.beta(1.0, 1.0)
    rand_idx = torch.randperm(x.size(0)).to(x.device)
    return ratio * x + (1 - ratio) * x[rand_idx], y, y[rand_idx], ratio


class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class LabelSmoothing(nn.Module):
    def __init__(self, alpha=0.1):
        super(LabelSmoothing, self).__init__()
        self.alpha = alpha
        self.certainty = 1.0 - alpha
        self.criterion = nn.KLDivLoss(reduction='mean')

    def forward(self, x, y):
        b, c = x.shape
        label = torch.full((b, c), self.alpha / (c - 1)).to(y.device)
        label = label.scatter(1, y.unsqueeze(1), self.certainty)
        return self.criterion(F.log_softmax(x, dim=1), label)


class RichDataSummary(Callback):
    def on_pretrain_routine_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_global_zero:
            console = Console()
            data = trainer.datamodule
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Dataset Name", justify="center")
            table.add_column("Num Class", justify="center")
            table.add_column("Batch Size", justify="right")
            table.add_column("Step", justify="right")
            table.add_column("Train", justify="right")
            table.add_column("Test", justify="right")
            table.add_row(
                data.dataset_name,
                str(data.num_classes),
                str(data.batch_size),
                str(data.num_step),
                str(data.train_data_len),
                str(data.test_data_len)
            )
            console.print(table)
