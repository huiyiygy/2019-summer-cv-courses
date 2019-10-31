# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: classes_net.py
@time: 2019/10/30 下午1:40
"""
import torch
import torch.nn as nn
from model.backbone.xception import xception
from model.backbone.inceptionv4 import inceptionv4


class ClassesNet(nn.Module):
    def __init__(self, backbone='xception', num_classes=2, pretrained=False, freeze_bn=False):
        super(ClassesNet, self).__init__()
        if backbone == 'xception':
            self.backbone = xception(num_classes, pretrained)
        elif backbone == 'inceptionv4':
            self.backbone = inceptionv4(num_classes, pretrained)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs):
        x = self.backbone(inputs)
        return x

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.Linear):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = ClassesNet(backbone='xception', pretrained=True)
    model.eval()
    inputs = torch.rand(1, 3, 299, 299)
    output = model(inputs)
    print(output.size())
