import os
import math
import random
import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.swa_utils import update_bn
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import instantiate_class, LightningCLI
from torchmetrics import MetricCollection, Accuracy

from utils import LabelSmoothing, cutmix, cutout, mixup, EMA
from data import BaseDataModule


class BaseVisionSystem(LightningModule):
    def __init__(self, model_name: str, pretrained: bool, num_classes: int, num_step: int, max_epochs: int,
                 gpus: str, optimizer_init: dict, lr_scheduler_init: dict, use_precise_bn: bool = False,
                 augmentation: str = 'default', ema: float = 0.0, dropout: float = 0.0):
        """ Define base vision classification system
        :arg
            model_name: model name string ex) efficientnet_v2_s
            pretrained: use pretrained weight or not
            num_classes: number of class of dataset
            num_step: number of step for 1 epoch
            max_epochs: number of epoch to train
            gpus: gpus id string ex) 1,2,
            optimizer_init: optimizer class path and init args
            lr_scheduler_init: learning rate scheduler class path and init args
            use_precise_bn: precise_bn is re-calculating batch statistic after each epoch end.
            augmentation: use mixup based augmentation ex) cutmix, cutout, mixup
            ema: use exponential moving average to increase model performance
            dropout: dropout rate for model
        """
        super(BaseVisionSystem, self).__init__()
        self.save_hyperparameters()

        # step 1. save data related info (not defined here)
        self.augmentation = augmentation
        self.gpus = len(gpus.split(',')) - 1
        self.num_step = int(math.ceil(num_step / (self.gpus)))
        self.max_epochs = max_epochs

        # step 2. define model
        self.model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', model_name, nclass=num_classes, skip_validation=True,
                                    pretrained=pretrained, dropout=dropout)
        self.ema_model = EMA(self.model, decay=ema) if ema != 0.0 else None
        self.use_precise_bn = use_precise_bn

        # step 3. define lr tools (optimizer, lr scheduler)
        self.optimizer_init_config = optimizer_init
        self.lr_scheduler_init_config = lr_scheduler_init
        self.criterion = LabelSmoothing()

        # step 4. define metric
        metrics = MetricCollection({'top@1': Accuracy(top_k=1), 'top@5': Accuracy(top_k=5)})
        self.train_metric = metrics.clone(prefix='train/')
        self.valid_metric = metrics.clone(prefix='valid/')
        self.test_metric = metrics.clone(prefix='test/')

    def forward(self, batch, batch_idx):
        x, y = batch
        loss, y_hat = self.compute_loss_eval(x, y)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.shared_step(batch, self.train_metric, 'train', add_dataloader_idx=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.valid_metric, 'valid', add_dataloader_idx=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.test_metric, 'test', add_dataloader_idx=True)

    def shared_step(self, batch, metric, mode, add_dataloader_idx):
        x, y = batch
        loss, y_hat = self.compute_loss(x, y) if mode == 'train' else self.compute_loss_eval(x, y)
        metric = metric(y_hat, y)
        self.log_dict({f'{mode}/loss': loss}, add_dataloader_idx=add_dataloader_idx, prog_bar=True)
        self.log_dict(metric, add_dataloader_idx=add_dataloader_idx, prog_bar=True)
        return loss

    def compute_loss(self, x, y):
        if random.random() < 1.0 and self.augmentation != 'default':
            if self.augmentation == 'mixup':
                x, y1, y2, ratio = mixup(x, y)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y1) * ratio + self.criterion(y_hat, y2) * (1 - ratio)
                return loss, y_hat

            elif self.augmentation == 'cutout':
                x, y, ratio = cutout(x, y)
                return self.compute_loss_eval(x, y)

            elif self.augmentation == 'cutmix':
                x, y1, y2, ratio = cutmix(x, y)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y1) * ratio + self.criterion(y_hat, y2) * (1 - ratio)
                return loss, y_hat
        else:
            return self.compute_loss_eval(x, y)

    def compute_loss_eval(self, x, y):
        y_hat = self.model(x) if self.model.training or self.ema_model is None else self.ema_model.module(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat

    def configure_optimizers(self):
        optimizer = instantiate_class(self.model.parameters(), self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def update_and_get_lr_scheduler_config(self):
        if 'num_step' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['num_step'] = self.num_step
        if 'max_epochs' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['max_epochs'] = self.max_epochs
        if 'max_lr' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['max_lr'] = self.optimizer_init_config['init_args']['lr']
        if 'total_steps' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['total_steps'] = self.num_step * self.max_epochs
        return self.lr_scheduler_init_config

    @torch.no_grad()
    def on_train_end(self) -> None:
        """"Precise BatchNorm"""
        if self.use_precise_bn:
            if self.ema_model:
                update_bn(self.trainer.datamodule.train_dataloader(), self.ema_model, self.device)
            else:
                update_bn(self.trainer.datamodule.train_dataloader(), self.model, self.device)

    def on_after_backward(self) -> None:
        """EMA model update"""
        if self.ema_model:
            self.ema_model.update(self.model)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        """Faster optimization step"""
        optimizer.zero_grad(set_to_none=True)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # 1. link argument
        parser.link_arguments('data.num_classes', 'model.num_classes', apply_on='instantiate')
        parser.link_arguments('data.num_step', 'model.num_step', apply_on='instantiate')
        parser.link_arguments('trainer.max_epochs', 'model.max_epochs', apply_on='parse')
        parser.link_arguments('trainer.gpus', 'model.gpus', apply_on='parse')

        # 2. add optimizer & scheduler argument
        parser.add_optimizer_args((SGD, Adam, AdamW), link_to='model.optimizer_init')
        parser.add_lr_scheduler_args((OneCycleLR,), link_to='model.lr_scheduler_init')


if __name__ == '__main__':
    cli = MyLightningCLI(BaseVisionSystem, BaseDataModule, save_config_overwrite=True)
    cli.trainer.test(ckpt_path='best', dataloaders=cli.datamodule.test_dataloader())
