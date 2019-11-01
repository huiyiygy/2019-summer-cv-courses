# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: multi_net.py
@time: 2019/10/30 下午1:42
"""
import torch
import torch.nn as nn
from model.backbone.xception import multi_xception


class MultiNet(nn.Module):
    def __init__(self, backbone='xception', num_classes=None, pretrained=False, freeze_bn=False):
        super(MultiNet, self).__init__()
        if backbone == 'xception':
            self.backbone = multi_xception(num_classes, pretrained)
        else:
            raise NotImplementedError

        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs):
        x_classes, x_species = self.backbone(inputs)
        return x_classes, x_species

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
    model = MultiNet(backbone='xception', num_classes={'classes_num': 2, 'species_num': 3}, pretrained=True)
    model.eval()
    inputs = torch.rand(1, 3, 299, 299)
    output_classes, output_species = model(inputs)
    print(output_classes.size())
    print(output_species.size())
