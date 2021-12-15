import json
import os

import math
import warnings

import torch
from torch.nn.utils.rnn import pack_padded_sequence

warnings.filterwarnings('ignore')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import instantiate_class, LightningCLI
from torchmetrics import MetricCollection, Accuracy, BLEUScore

from utils import LabelSmoothing, EMA
from data import BaseDataModule
from model import get_model


class BaseImageCaptionSystem(LightningModule):
    def __init__(self, model_name: str, pretrained: bool, num_step: int, max_epochs: int,
                 gpus: str, optimizer_init: dict, lr_scheduler_init: dict,
                 processed_root: str, dropout: float = 0.0, save_folder='inference_results'):
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
        super(BaseImageCaptionSystem, self).__init__()
        self.save_hyperparameters()

        # step 1. save data related info (not defined here)
        self.gpus = len(gpus.split(',')) - 1
        self.num_step = int(math.ceil(num_step / (self.gpus)))
        self.max_epochs = max_epochs

        # step 2. define model
        self.word_map = self.open_word_map(processed_root)
        self.idx_map = {v: k for k, v in self.word_map.items()}
        self.model = get_model(model_name, pretrained, len(self.word_map), dropout)

        # step 3. define lr tools (optimizer, lr scheduler)
        self.optimizer_init_config = optimizer_init
        self.lr_scheduler_init_config = lr_scheduler_init
        self.criterion = LabelSmoothing()

        # step 4. define metric
        metrics = MetricCollection({'top@1': Accuracy(top_k=1), 'top@5': Accuracy(top_k=5)})
        self.train_metric = metrics.clone(prefix='train/')
        self.valid_metric = metrics.clone(prefix='valid/')
        self.test_metric = metrics.clone(prefix='test/')
        self.bleu_metric = BLEUScore()
        self.beam_bleu_metric = BLEUScore()
        self.save_folder = save_folder
        self.results = {'references': [], 'hypothesis': []}

    def forward(self, batch, batch_idx):
        x, y = batch
        loss, y_hat = self.compute_loss_eval(x, y)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.shared_step(batch, self.train_metric, 'train')

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch[:-1], self.valid_metric, 'valid', batch[-1], self.bleu_metric)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        imgs, _, _, references = batch
        references = self.get_reference_list(references, list(range(len(references))))
        hypothesis = [self.model.inference(img, self.word_map, self.idx_map) for img in imgs]
        self.log_dict({f'test/BLEU@4': self.beam_bleu_metric(references, hypothesis)}, prog_bar=True)
        self.show_example(imgs[0], references[0][0], hypothesis[0], batch_idx)
        self.results['hypothesis'].extend(hypothesis)
        self.results['references'].extend(references)

    def on_test_end(self) -> None:
        import json
        with open(f'{self.save_folder}/inference_results.json', 'w') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)

    def show_example(self, img, reference, hypothesis, batch_idx):
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        Path(self.save_folder).mkdir(exist_ok=True)
        plt.figure(figsize=(16,20))
        mean, std = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]), torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
        img = np.transpose((img.clone().cpu() * std + mean).numpy(), (1, 2, 0))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f'{hypothesis}\n{reference}', fontsize=18)
        plt.savefig(f'{self.save_folder}/{batch_idx}.png')

    def shared_step(self, batch, metric, mode, references=None, bleu_metric=None):
        preds, caps_sorted, decode_lengths, alphas, sort_ind = self.model(*batch)
        y_hat = pack_padded_sequence(preds, decode_lengths, batch_first=True).data
        y = pack_padded_sequence(caps_sorted[:, 1:], decode_lengths, batch_first=True).data
        loss = self.criterion(y_hat, y) + ((1. - alphas.sum(dim=1)) ** 2).mean()
        metric = metric(y_hat, y)
        self.log_dict({f'{mode}/loss': loss}, prog_bar=True)
        self.log_dict(metric, prog_bar=True)

        if bleu_metric:
            references = self.get_reference_list(references, sort_ind)
            hypothesis = [' '.join(self.idx_map[i] for i in pred[:decode_lengths[j]]) for j, pred in enumerate(torch.max(preds, dim=2)[1].tolist())]
            self.log_dict({f'{mode}/teaching_force_BLEU@4': bleu_metric(references, hypothesis)}, prog_bar=True)

        return loss

    def get_reference_list(self, references, sort_ind):
        reference_list = []
        references = references[sort_ind]
        for idx in range(references.size(0)):
            reference = references[idx].tolist()
            reference = list(
                map(lambda c: ' '.join([self.idx_map[w] for w in c if w not in {self.word_map['<start>'], self.word_map['<end>'], self.word_map['<pad>']}]), reference))
            reference_list.append(reference)
        return reference_list

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

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        """Faster optimization step"""
        optimizer.zero_grad(set_to_none=True)

    def open_word_map(self, processed_root):
        with open(os.path.join(processed_root, 'WORDMAP.json'), 'r') as f:
            word_map = json.load(f)
        return word_map


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # 1. link argument
        parser.link_arguments('data.processed_root', 'model.processed_root', apply_on='instantiate')
        parser.link_arguments('data.num_step', 'model.num_step', apply_on='instantiate')
        parser.link_arguments('trainer.max_epochs', 'model.max_epochs', apply_on='parse')
        parser.link_arguments('trainer.gpus', 'model.gpus', apply_on='parse')

        # 2. add optimizer & scheduler argument
        parser.add_optimizer_args((SGD, Adam, AdamW), link_to='model.optimizer_init')
        parser.add_lr_scheduler_args((OneCycleLR,), link_to='model.lr_scheduler_init')


if __name__ == '__main__':
    cli = MyLightningCLI(BaseImageCaptionSystem, BaseDataModule, save_config_overwrite=True)
    # cli.trainer.test(ckpt_path='best', dataloaders=cli.datamodule.test_dataloader())