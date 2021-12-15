import math
import os
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from random import seed, choice, sample

import cv2
import h5py
import numpy as np
from PIL import Image
from rich.progress import track

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torchvision import transforms


def create_input_files(downloaded_folder, output_folder, captions_per_image=3, min_word_freq=2, max_len=100):
    if Path(output_folder).exists():
        print('input files are already processed')
        return output_folder
    else:
        Path(output_folder).mkdir()

    with open(os.path.join(downloaded_folder, 'dataset_coco.json'), 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(downloaded_folder, img['filepath'], img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        save_caption_with_img(captions_per_image, imcaps, impaths, max_len, output_folder, split, word_map)

    return output_folder


def save_caption_with_img(captions_per_image, imcaps, impaths, max_len, output_folder, split, word_map):
    with h5py.File(os.path.join(output_folder, split + '_IMAGES.hdf5'), 'a') as h:
        # Make a note of the number of captions we are sampling per image
        h.attrs['captions_per_image'] = captions_per_image

        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

        enc_captions = []
        caplens = []
        for i, path in track(enumerate(impaths), total=len(impaths), description=split):
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)

            # Sanity check
            assert len(captions) == captions_per_image

            # Read images
            img = np.asarray(Image.open(impaths[i]))
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = cv2.resize(img, (256, 256))
            img = img.transpose(2, 0, 1)
            assert np.max(img) <= 255

            # Save image to HDF5 file
            images[i] = img

            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                # Find caption lengths
                c_len = len(c) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)

        # Sanity check
        assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

        # Save encoded captions and their lengths to JSON files
        with open(os.path.join(output_folder, split + '_CAPTIONS.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder, split + '_CAPLENS.json'), 'w') as j:
            json.dump(caplens, j)


def download_ms_coco_2014(data_root):
    Path(data_root).mkdir(exist_ok=True, parents=True)
    extracted_files = ['train2014', 'val2014', 'test2014', 'annotations', 'dataset_coco.json']
    is_extracted = all(Path(os.path.join(data_root, file)).exists() for file in extracted_files)
    if is_extracted:
        print('ms_coco_2014 is downloaded and extracted already')
        return

    url_list = [
        "http://images.cocodataset.org/zips/train2014.zip",
        "http://images.cocodataset.org/zips/val2014.zip",
        "http://images.cocodataset.org/zips/test2014.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
    ]
    file_names = [os.path.join(data_root, path.split('/')[-1]) for path in url_list]
    for filename, url in zip(file_names, url_list):
        if Path(filename).exists():
            print(f'{filename} data already downloaded')
        else:
            subprocess.run(["wget", "-r", "-nc", '-O', filename, url])
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(data_root)


class MSCOCO2014(Dataset):
    def __init__(self, root, split='TRAIN', download=False, captions_per_image=3, min_word_freq=2, transform=None):
        self.processed_root = os.path.join(root, f'{captions_per_image}_cap_per_img_{min_word_freq}_min_word_freq')

        if download == True:
            download_ms_coco_2014(root)
            create_input_files(root, self.processed_root, captions_per_image, min_word_freq)

        self.split = split.upper()
        self.train = self.split == 'TRAIN'

        self.h = h5py.File(os.path.join(self.processed_root, self.split + '_IMAGES.hdf5'), 'r')
        self.imgs = self.h['images']
        self.cpi = self.h.attrs['captions_per_image']

        with open(os.path.join(self.processed_root, self.split + '_CAPTIONS.json'), 'r') as j:
            self.captions = json.load(j)

        with open(os.path.join(self.processed_root, self.split + '_CAPLENS.json'), 'r') as j:
            self.caplens = json.load(j)

        self.transform = transform
        self.dataset_size = len(self.captions)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        img_idx = i // self.cpi
        img = torch.FloatTensor(self.imgs[img_idx] / 255.)
        caption, caplen = torch.LongTensor(self.captions[i]), torch.LongTensor([self.caplens[i]])

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            base_caption_idx = img_idx * self.cpi
            all_captions = torch.LongTensor(self.captions[base_caption_idx:(base_caption_idx + self.cpi)])
            return img, caption, caplen, all_captions


class BaseDataModule(LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 size: tuple,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 data_root: str = '/home/hankyul/hdd_ext2/coco',
                 captions_per_image=5,
                 min_word_freq=5):
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

        if dataset_name == 'coco':
            dataset, self.mean, self.std = MSCOCO2014, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        self.size = self.eval_size = size
        self.train_transform, self.test_transform = self.get_transforms()
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.captions_per_image = captions_per_image
        self.min_word_freq = min_word_freq
        self.data_processed_name = None
        self.num_classes = None
        self.num_step = None
        self.train_data_len = None
        self.valid_data_len = None
        self.test_data_len = None
        self.prepare_data()

    def get_transforms(self):
        train = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        test = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        return train, test

    def prepare_data(self) -> None:
        train = self.dataset(self.data_root, 'train', True, self.captions_per_image, self.min_word_freq)
        valid = self.dataset(self.data_root, 'val', True, self.captions_per_image, self.min_word_freq)
        test = self.dataset(self.data_root, 'test', True, self.captions_per_image, self.min_word_freq)

        self.processed_root = train.processed_root
        self.train_data_len = len(train)
        self.valid_data_len = len(valid)
        self.test_data_len = len(test)
        self.num_step = int(math.ceil(len(train) / self.batch_size))

    def setup(self, stage: str = None):
        self.train_ds = self.dataset(self.data_root, 'train', False, self.captions_per_image, self.min_word_freq, self.train_transform)
        self.valid_ds = self.dataset(self.data_root, 'val', False, self.captions_per_image, self.min_word_freq, self.test_transform)
        self.test_ds = self.dataset(self.data_root, 'test', False, self.captions_per_image, self.min_word_freq, self.test_transform)

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