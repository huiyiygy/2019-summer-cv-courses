# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: __init__.py.py
@time: 2019/10/30 上午10:15
"""
from dataloader.dataset.my_dataset import MyDataset
from torch.utils.data import DataLoader

ROOT_DIR = '/home/lab/ygy/cv_course_project/2019-summer-cv-courses/project_1_Species_classification'

CLASSES_TRAIN_ANNO = '/home/lab/ygy/cv_course_project/2019-summer-cv-courses/project_1_Species_classification/doc/Stage_1_Classes_classification/Classes_train_annotation.csv'
CLASSES_VAL_ANNO = '/home/lab/ygy/cv_course_project/2019-summer-cv-courses/project_1_Species_classification/doc/Stage_1_Classes_classification/Classes_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']

SPECIES_TRAIN_ANNO = '/home/lab/ygy/cv_course_project/2019-summer-cv-courses/project_1_Species_classification/doc/Stage_2_Species_classification/Species_train_annotation.csv'
SPECIES_VAL_ANNO = '/home/lab/ygy/cv_course_project/2019-summer-cv-courses/project_1_Species_classification/doc/Stage_2_Species_classification/Species_val_annotation.csv'
SPECIES = ['rabbits', 'rats', 'chickens']

MULTI_TRAIN_ANNO = '/home/lab/ygy/cv_course_project/2019-summer-cv-courses/project_1_Species_classification/doc/Stage_3_Multi-classification/Multi_train_annotation.csv'
MULTI_VAL_ANNO = '/home/lab/ygy/cv_course_project/2019-summer-cv-courses/project_1_Species_classification/doc/Stage_3_Multi-classification/Multi_val_annotation.csv'


def make_data_loader(args, **kwargs):
    if args.dataset == 'Classes':
        train_set = MyDataset(args, ROOT_DIR, CLASSES_TRAIN_ANNO, split="train")
        val_set = MyDataset(args, ROOT_DIR, CLASSES_VAL_ANNO, split="val")
        num_class = len(CLASSES)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'Species':
        train_set = MyDataset(args, ROOT_DIR, SPECIES_TRAIN_ANNO, split="train")
        val_set = MyDataset(args, ROOT_DIR, SPECIES_VAL_ANNO, split="val")
        num_class = len(SPECIES)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'Multi':
        train_set = MyDataset(args, ROOT_DIR, MULTI_TRAIN_ANNO, split="train")
        val_set = MyDataset(args, ROOT_DIR, MULTI_VAL_ANNO, split="val")
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        num_class = {'classes_num': len(CLASSES), 'species_num': len(SPECIES)}
    else:
        raise NotImplementedError

    return train_loader, val_loader, num_class
