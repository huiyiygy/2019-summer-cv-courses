# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: my_dataset.py
@time: 2019/10/30 下午1:55
"""
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import dataloader.custom_transforms as tr


class MyDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, args, root_dir, annotations_file, split="train"):
        self.args = args
        self.dataset = args.dataset
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.split = split

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        image_path = image_path.replace('\\', '/')
        image_path = os.path.join(self.root_dir, image_path[3:])
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None

        image = Image.open(image_path).convert('RGB')
        label_class = None
        if self.dataset == 'Classes':
            label_class = int(self.file_info.iloc[idx]['classes'])
        elif self.dataset == 'Species':
            label_class = int(self.file_info.iloc[idx]['species'])

        sample = {'image': image, 'label': label_class}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomVerticalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomRotate(),
            tr.RandomGaussianBlur(),
            tr.RandomNoise(),
            # mean and std calculated by calculateMeanStd.py
            tr.Normalize(mean=(0.656963, 0.621670, 0.550278), std=(0.300198, 0.303201, 0.334976)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.656963, 0.621670, 0.550278), std=(0.300198, 0.303201, 0.334976)),
            tr.ToTensor()])

        return composed_transforms(sample)
