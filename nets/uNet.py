#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision import models
from nets import block

class uNet(nn.Module):
    
    def __init__(self, n_channels, n_classes):
        
        super(uNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.enblock1 = block.DoubleConv(self.n_channels, 32)
        self.enblock2 = block.DoubleDown(32, 64)
        self.enblock3 = block.DoubleDown(64, 128)
        self.enblock4 = block.DoubleDown(128, 256)
        self.center = block.DoubleDown(256, 256)
        self.deblock1 = block.DoubleUp(256, 256, 512)
        self.deblock2 = block.DoubleUp(256, 128)
        self.deblock3 = block.DoubleUp(128, 64)
        self.deblock4 = block.DoubleUp(64, 32)
        self.outc = block.OutConv(32, self.n_classes)

    def forward(self, x):
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        x = self.deblock1(x5,x4)
        x = self.deblock2(x,x3)
        x = self.deblock3(x,x2)
        x = self.deblock4(x,x1)
        logits = self.outc(x)
        return logits
    
class uNet13(nn.Module):
    
    def __init__(self, n_classes, pretrained=True):    
        super(uNet13, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg13_bn(weights='VGG13_BN_Weights.DEFAULT').features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[21], self.encoder[22], self.relu, self.encoder[24], self.encoder[25], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[28], self.encoder[29], self.relu, self.encoder[31], self.encoder[32], self.relu)
        self.deblock1 = block.DoubleUp(512,512,1024)
        self.deblock2 = block.DoubleUp(512,256)
        self.deblock3 = block.DoubleUp(256,128)
        self.deblock4 = block.DoubleUp(128,64)
        self.outc = block.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        x = self.deblock1(x5, x4)
        x = self.deblock2(x, x3)
        x = self.deblock3(x, x2)
        x = self.deblock4(x, x1)
        logits = self.outc(x)
        return logits

class uNet16(nn.Module):
    
    def __init__(self, n_classes, pretrained=True):    
        super(uNet16, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg16_bn(weights='VGG16_BN_Weights.DEFAULT').features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu, self.encoder[20], self.encoder[21], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[24], self.encoder[25], self.relu, self.encoder[27], self.encoder[28], self.relu, self.encoder[30], self.encoder[31], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[34], self.encoder[35], self.relu, self.encoder[37], self.encoder[38], self.relu, self.encoder[40], self.encoder[41], self.relu)
        self.deblock1 = block.TripleUp(512,512,1024)
        self.deblock2 = block.TripleUp(512,256)
        self.deblock3 = block.DoubleUp(256,128)
        self.deblock4 = block.DoubleUp(128,64)
        self.outc = block.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        x = self.deblock1(x5, x4)
        x = self.deblock2(x, x3)
        x = self.deblock3(x, x2)
        x = self.deblock4(x, x1)
        logits = self.outc(x)
        return logits

class uNet19(nn.Module):
    
    def __init__(self, n_classes, pretrained=True):    
        super(uNet19, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT').features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu, self.encoder[20], self.encoder[21], self.relu, self.encoder[23], self.encoder[24], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[27], self.encoder[28], self.relu, self.encoder[30], self.encoder[31], self.relu, self.encoder[33], self.encoder[34], self.relu, self.encoder[36], self.encoder[37], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[40], self.encoder[41], self.relu, self.encoder[43], self.encoder[44], self.relu, self.encoder[46], self.encoder[47], self.relu, self.encoder[49], self.encoder[50], self.relu)
        self.deblock1 = block.QuadripleUp(512,512,1024)
        self.deblock2 = block.QuadripleUp(512,256)
        self.deblock3 = block.DoubleUp(256,128)
        self.deblock4 = block.DoubleUp(128,64)
        self.outc = block.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        x = self.deblock1(x5, x4)
        x = self.deblock2(x, x3)
        x = self.deblock3(x, x2)
        x = self.deblock4(x, x1)
        logits = self.outc(x)
        return logits