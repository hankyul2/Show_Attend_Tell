import math
import random
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import Callback

import numpy as np
from rich.table import Table
from rich.console import Console


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
