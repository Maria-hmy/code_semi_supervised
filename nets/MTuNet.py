#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision import models
from nets import block
import numpy as np

class MTuNet19(nn.Module): # MT = multi-tasks
    
    def __init__(self, n_classes, pretrained=True):    
        super(MTuNet19, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes

        # == single encoder
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg19_bn(pretrained=pretrained).features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu, self.encoder[20], self.encoder[21], self.relu, self.encoder[23], self.encoder[24], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[27], self.encoder[28], self.relu, self.encoder[30], self.encoder[31], self.relu, self.encoder[33], self.encoder[34], self.relu, self.encoder[36], self.encoder[37], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[40], self.encoder[41], self.relu, self.encoder[43], self.encoder[44], self.relu, self.encoder[46], self.encoder[47], self.relu, self.encoder[49], self.encoder[50], self.relu)
        # == first decoder
        self.D1_deblock1 = block.QuadripleUp(512,512,1024)
        self.D1_deblock2 = block.QuadripleUp(512,256)
        self.D1_deblock3 = block.DoubleUp(256,128)
        self.D1_deblock4 = block.DoubleUp(128,64)
        self.D1_outc = block.OutConv(64, self.n_classes)
        # == second decoder
        self.D2_deblock1 = block.QuadripleUp(512,512,1024)
        self.D2_deblock2 = block.QuadripleUp(512,256)
        self.D2_deblock3 = block.DoubleUp(256,128)
        self.D2_deblock4 = block.DoubleUp(128,64)
        self.D2_outc = block.OutConv(64, self.n_classes)
        

    def forward(self, x):
        # == single encoder
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        # == first decoder
        D1_x = self.D1_deblock1(x5, x4)
        D1_x = self.D1_deblock2(D1_x, x3)
        D1_x = self.D1_deblock3(D1_x, x2)
        D1_x = self.D1_deblock4(D1_x, x1)
        D1_logits = self.D1_outc(D1_x)
        # == second decoder
        D2_x = self.D2_deblock1(x5, x4)
        D2_x = self.D2_deblock2(D2_x, x3)
        D2_x = self.D2_deblock3(D2_x, x2)
        D2_x = self.D2_deblock4(D2_x, x1)
        D2_logits = self.D2_outc(D2_x)
        
        return D1_logits, D2_logits
    
class MTcuNet19(nn.Module):
    
    def __init__(self, n_classes, in_size=256, pretrained=True):    
        super(MTcuNet19, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.in_size = in_size

        # == single encoder
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg19_bn(pretrained=pretrained).features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu, self.encoder[20], self.encoder[21], self.relu, self.encoder[23], self.encoder[24], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[27], self.encoder[28], self.relu, self.encoder[30], self.encoder[31], self.relu, self.encoder[33], self.encoder[34], self.relu, self.encoder[36], self.encoder[37], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[40], self.encoder[41], self.relu, self.encoder[43], self.encoder[44], self.relu, self.encoder[46], self.encoder[47], self.relu, self.encoder[49], self.encoder[50], self.relu)
        # == first decoder
        self.D1_deblock1 = block.QuadripleUp(512,512,1024)
        self.D1_deblock2 = block.QuadripleUp(512,256)
        self.D1_deblock3 = block.DoubleUp(256,128)
        self.D1_deblock4 = block.DoubleUp(128,64)
        self.D1_outc = block.OutConv(64, self.n_classes)
        # == second decoder
        self.D2_deblock1 = block.QuadripleUp(512,512,1024)
        self.D2_deblock2 = block.QuadripleUp(512,256)
        self.D2_deblock3 = block.DoubleUp(256,128)
        self.D2_deblock4 = block.DoubleUp(128,64)
        self.D2_outc = block.OutConv(64, self.n_classes)
        """
        # == classification branch
        self.clf_block1 = block.DoubleConv(512,64)
        self.clf_block2 = block.DoubleConv(64,4)
        size_ = np.int(self.in_size/16)
        self.clf_block3 = nn.Linear(4*size_*size_,128)
        self.clf_block4 = nn.ReLU()
        self.clf_block5 = nn.Linear(128,1)
        """

    def forward(self, x):
        # == single encoder
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        # == first decoder
        D1_x = self.D1_deblock1(x5, x4)
        D1_x = self.D1_deblock2(D1_x, x3)
        D1_x = self.D1_deblock3(D1_x, x2)
        D1_x = self.D1_deblock4(D1_x, x1)
        D1_logits = self.D1_outc(D1_x)
        # == second decoder
        D2_x = self.D2_deblock1(x5, x4)
        D2_x = self.D2_deblock2(D2_x, x3)
        D2_x = self.D2_deblock3(D2_x, x2)
        D2_x = self.D2_deblock4(D2_x, x1)
        D2_logits = self.D2_outc(D2_x)
        # == classification branch
        """
        cl_x = self.clf_block1(x5)
        cl_x = self.clf_block2(cl_x)
        cl_x = cl_x.view(cl_x.size(0), -1)
        cl_x = self.clf_block3(cl_x)
        cl_x = self.clf_block4(cl_x)
        cl_x = self.clf_block5(cl_x)
        """
        
        return D1_logits, D2_logits, x5
