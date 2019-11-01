# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: train.py
@time: 2019/10/28 下午8:08
"""
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm

from dataloader import make_data_loader
from model.classes_net import ClassesNet
from model.multi_net import MultiNet
from utils.saver import Saver
from tensorboardX import SummaryWriter


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.writer = SummaryWriter(log_dir=self.saver.experiment_dir)

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.nclass = make_data_loader(args, **kwargs)

        model = None
        # Define network
        if self.args.dataset != 'Multi':
            model = ClassesNet(backbone=self.args.backbone, num_classes=self.nclass, pretrained=True)
            if self.args.dataset == 'Classes':
                print("Training ClassesNet")
            else:
                print("Training SpeciesNet")
        else:
            model = MultiNet(backbone=self.args.backbone, num_classes=self.nclass, pretrained=True)
            print("Training MultiNet")

        self.model = model

        train_params = [{'params': model.get_params()}]
        # Define Optimizer
        self.optimizer = torch.optim.Adam(train_params, self.args.learn_rate, weight_decay=args.weight_decay, amsgrad=args.nesterov)

        # Define Criterion
        self.criterion = nn.CrossEntropyLoss(size_average=True)

        # Define lr scheduler
        exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        print('[Epoch: %d, learning rate: %.6f, previous best = %.4f]' % (epoch, self.args.learn_rate, self.best_pred))
        train_loss = 0.0
        corrects_labels = 0
        correct_classes = 0
        correct_species = 0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            self.optimizer.zero_grad()
            if self.args.dataset != 'Multi':
                image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                output = self.model(image)
                loss = self.criterion(output, target)

                pred_label = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred_label = np.argmax(pred_label, axis=1)
                corrects_labels += np.sum(pred_label == target)

            else:
                image, target_classes, target_species = sample['image'], sample['classes_label'], sample['species_label']
                if self.args.cuda:
                    image, target_classes, target_species = image.cuda(), target_classes.cuda(), target_species.cuda()
                output_classes, output_species = self.model(image)
                classes_loss = self.criterion(output_classes, target_classes)
                species_loss = self.criterion(output_species, target_species)
                loss = classes_loss + species_loss

                pred_classes = output_classes.data.cpu().numpy()
                pred_species = output_species.data.cpu().numpy()
                target_classes = target_classes.data.cpu().numpy()
                target_species = target_species.data.cpu().numpy()
                pred_classes = np.argmax(pred_classes, axis=1)
                pred_species = np.argmax(pred_species, axis=1)

                tmp1 = pred_classes == target_classes
                tmp2 = target_species == pred_species
                correct_classes += np.sum(tmp1)  # 统计“纲”分类正确的数量
                correct_species += np.sum(tmp2)  # 统计“种”分类正确的数量
                corrects_labels += np.sum(tmp1 & tmp2)  # 按位与，统计“纲”、“种”同时分类正确的数量

            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            train_loss += loss.item()
            tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))

        # Fast test during the training
        acc = corrects_labels / len(self.train_loader.dataset)
        self.writer.add_scalar('train/Acc', acc, epoch)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        acc_classes, acc_species = 0.0, 0.0
        if self.args.dataset == 'Multi':
            acc_classes = correct_classes / len(self.train_loader.dataset)
            acc_species = correct_species / len(self.train_loader.dataset)
            self.writer.add_scalar('train/Acc_classes', acc_classes, epoch)
            self.writer.add_scalar('train/Acc_species', acc_species, epoch)

        print('train validation:')
        if self.args.dataset != 'Multi':
            print("Acc:{}".format(acc))
        else:
            print("Acc:{}, Acc_classes:{}, Acc_species:{}".format(acc, acc_classes, acc_species))
        print('Loss: %.5f' % train_loss)
        print('---------------------------------')

    def validation(self, epoch):
        test_loss = 0.0
        corrects_labels = 0
        correct_classes = 0
        correct_species = 0
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        num_img_val = len(self.val_loader)

        for i, sample in enumerate(tbar):

            if self.args.dataset != 'Multi':
                image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    output = self.model(image)

                loss = self.criterion(output, target)

                pred_label = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred_label = np.argmax(pred_label, axis=1)
                corrects_labels += np.sum(pred_label == target)
            else:
                image, target_classes, target_species = sample['image'], sample['classes_label'], sample['species_label']
                if self.args.cuda:
                    image, target_classes, target_species = image.cuda(), target_classes.cuda(), target_species.cuda()
                with torch.no_grad():
                    output_classes, output_species = self.model(image)

                classes_loss = self.criterion(output_classes, target_classes)
                species_loss = self.criterion(output_species, target_species)
                loss = classes_loss + species_loss

                pred_classes = output_classes.data.cpu().numpy()
                pred_species = output_species.data.cpu().numpy()
                target_classes = target_classes.data.cpu().numpy()
                target_species = target_species.data.cpu().numpy()
                pred_classes = np.argmax(pred_classes, axis=1)
                pred_species = np.argmax(pred_species, axis=1)

                tmp1 = pred_classes == target_classes
                tmp2 = target_species == pred_species
                correct_classes += np.sum(tmp1)  # 统计“纲”分类正确的数量
                correct_species += np.sum(tmp2)  # 统计“种”分类正确的数量
                corrects_labels += np.sum(tmp1 & tmp2)  # 按位与，统计“纲”、“种”同时分类正确的数量

            test_loss += loss.item()
            tbar.set_description('Test loss: %.5f' % (test_loss / (i + 1)))
            self.writer.add_scalar('val/total_loss_iter', loss.item(), i + num_img_val * epoch)

        # Fast test during the training
        acc = corrects_labels / len(self.val_loader.dataset)
        self.writer.add_scalar('val/Acc', acc, epoch)
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        acc_classes, acc_species = 0.0, 0.0
        if self.args.dataset == 'Multi':
            acc_classes = correct_classes / len(self.val_loader.dataset)
            acc_species = correct_species / len(self.val_loader.dataset)
            self.writer.add_scalar('val/Acc_classes', acc_classes, epoch)
            self.writer.add_scalar('val/Acc_species', acc_species, epoch)

        print('test validation:')
        if self.args.dataset != 'Multi':
            print("Acc:{}".format(acc))
        else:
            print("Acc:{}, Acc_classes:{}, Acc_species:{}".format(acc, acc_classes, acc_species))
        print('Loss: %.5f' % test_loss)
        print('====================================')

        new_pred = acc
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['xception', 'inceptionv4'],
                        help='backbone name (default: xception)')
    parser.add_argument('--dataset', type=str, default='Classes',
                        choices=['Classes', 'Species', 'Multi'],
                        help='dataset name (default: Classes)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=300,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=299,
                        help='crop image size')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                        help='input batch size for training (default: auto)')
    # optimizer params
    parser.add_argument('--learn-rate', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {'classes': 100, 'species': 100, 'multi': 100}
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.learn_rate is None:
        lrs = {'classes': 0.001, 'species': 0.001, 'multi': 0.001}
        args.learn_rate = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = str(args.backbone)

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    print('====================================')
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
