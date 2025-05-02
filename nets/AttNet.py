#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import models
from torch import nn
from nets import block

# derived from https://github.com/wulalago/FetalCPSeg/blob/6cb93a7ab86cd2664f1a38229608bb68fac5791b/Network/MixAttNet.py#L107

class uNet13Att(nn.Module):
    
    def __init__(self, n_classes, pretrained=True, attention=True):    
        super(uNet13Att, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.attention = attention
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg13_bn(pretrained=pretrained).features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[21], self.encoder[22], self.relu, self.encoder[24], self.encoder[25], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[28], self.encoder[29], self.relu, self.encoder[31], self.encoder[32], self.relu)
        self.deblock1 = block.DoubleUp(512,512,1024)
        self.deblock2 = block.DoubleUp(512,256)
        self.deblock3 = block.DoubleUp(256,128)
        self.deblock4 = block.DoubleUp(128,64)

        self.down4 = block.SimpleConv(512, 16)
        self.down3 = block.SimpleConv(256, 16)
        self.down2 = block.SimpleConv(128, 16)
        self.down1 = block.SimpleConv(64, 16)
        
        self.down_out1 = block.SingleConv(16, self.n_classes)
        self.down_out2 = block.SingleConv(16, self.n_classes)
        self.down_out3 = block.SingleConv(16, self.n_classes)
        self.down_out4 = block.SingleConv(16, self.n_classes)
        
        if self.attention:
            self.mix1 = block.Attention(16, 16)
            self.mix2 = block.Attention(16, 16)
            self.mix3 = block.Attention(16, 16)
            self.mix4 = block.Attention(16, 16)
        
            self.mix_out1 = block.SingleConv(16, self.n_classes)
            self.mix_out2 = block.SingleConv(16, self.n_classes)
            self.mix_out3 = block.SingleConv(16, self.n_classes)
            self.mix_out4 = block.SingleConv(16, self.n_classes)
        
        self.out1 = block.SimpleConv(16*4, 64)
        self.out2 = block.SingleConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        dec4 = self.deblock1(x5, x4)
        dec3 = self.deblock2(dec4, x3)
        dec2 = self.deblock3(dec3, x2)
        dec1 = self.deblock4(dec2, x1)
        
        down1 = block.up_sample2d(self.down1(dec1), x)
        down2 = block.up_sample2d(self.down2(dec2), x)
        down3 = block.up_sample2d(self.down3(dec3), x)
        down4 = block.up_sample2d(self.down4(dec4), x)

        down_out1 = self.down_out1(down1)
        down_out2 = self.down_out2(down2)
        down_out3 = self.down_out3(down3)
        down_out4 = self.down_out4(down4)

        if self.attention:
            mix1, att1 = self.mix1(down1)
            mix2, att2 = self.mix2(down2)
            mix3, att3 = self.mix3(down3)
            mix4, att4 = self.mix4(down4)
            mix_out1 = self.mix_out1(mix1)
            mix_out2 = self.mix_out2(mix2)
            mix_out3 = self.mix_out3(mix3)
            mix_out4 = self.mix_out4(mix4)
            logits = self.out2(self.out1(torch.cat((mix1, mix2, mix3, mix4), dim=1)))
        else:
            logits = self.out2(self.out1(torch.cat((down1, down2, down3, down4), dim=1)))
        
        if self.training:
            if self.attention: 
                return logits, mix_out1, mix_out2, mix_out3, mix_out4, down_out1, down_out2, down_out3, down_out4
            else:
                return logits, down_out1, down_out2, down_out3, down_out4
        else:
            return logits
        
class uNet16Att(nn.Module):
    
    def __init__(self, n_classes, pretrained=True, attention=True):    
        super(uNet16Att, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.attention = attention
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg16_bn(pretrained=pretrained).features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu, self.encoder[20], self.encoder[21], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[24], self.encoder[25], self.relu, self.encoder[27], self.encoder[28], self.relu, self.encoder[30], self.encoder[31], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[34], self.encoder[35], self.relu, self.encoder[37], self.encoder[38], self.relu, self.encoder[40], self.encoder[41], self.relu)
        self.deblock1 = block.TripleUp(512,512,1024)
        self.deblock2 = block.TripleUp(512,256)
        self.deblock3 = block.DoubleUp(256,128)
        self.deblock4 = block.DoubleUp(128,64)

        self.down4 = block.SimpleConv(512, 16)
        self.down3 = block.SimpleConv(256, 16)
        self.down2 = block.SimpleConv(128, 16)
        self.down1 = block.SimpleConv(64, 16)
        
        self.down_out1 = block.SingleConv(16, self.n_classes)
        self.down_out2 = block.SingleConv(16, self.n_classes)
        self.down_out3 = block.SingleConv(16, self.n_classes)
        self.down_out4 = block.SingleConv(16, self.n_classes)
        
        if self.attention:
            self.mix1 = block.Attention(16, 16)
            self.mix2 = block.Attention(16, 16)
            self.mix3 = block.Attention(16, 16)
            self.mix4 = block.Attention(16, 16)
        
            self.mix_out1 = block.SingleConv(16, self.n_classes)
            self.mix_out2 = block.SingleConv(16, self.n_classes)
            self.mix_out3 = block.SingleConv(16, self.n_classes)
            self.mix_out4 = block.SingleConv(16, self.n_classes)
        
        self.out1 = block.SimpleConv(16*4, 64)
        self.out2 = block.SingleConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        dec4 = self.deblock1(x5, x4)
        dec3 = self.deblock2(dec4, x3)
        dec2 = self.deblock3(dec3, x2)
        dec1 = self.deblock4(dec2, x1)
        
        down1 = block.up_sample2d(self.down1(dec1), x)
        down2 = block.up_sample2d(self.down2(dec2), x)
        down3 = block.up_sample2d(self.down3(dec3), x)
        down4 = block.up_sample2d(self.down4(dec4), x)

        down_out1 = self.down_out1(down1)
        down_out2 = self.down_out2(down2)
        down_out3 = self.down_out3(down3)
        down_out4 = self.down_out4(down4)

        if self.attention:
            mix1, att1 = self.mix1(down1)
            mix2, att2 = self.mix2(down2)
            mix3, att3 = self.mix3(down3)
            mix4, att4 = self.mix4(down4)
            mix_out1 = self.mix_out1(mix1)
            mix_out2 = self.mix_out2(mix2)
            mix_out3 = self.mix_out3(mix3)
            mix_out4 = self.mix_out4(mix4)
            logits = self.out2(self.out1(torch.cat((mix1, mix2, mix3, mix4), dim=1)))
        else:
            logits = self.out2(self.out1(torch.cat((down1, down2, down3, down4), dim=1)))
        
        if self.training:
            if self.attention: 
                return logits, mix_out1, mix_out2, mix_out3, mix_out4, down_out1, down_out2, down_out3, down_out4
            else:
                return logits, down_out1, down_out2, down_out3, down_out4
        else:
            return logits
        
class uNet19Att(nn.Module):
    
    def __init__(self, n_classes, pretrained=True, attention=True):    
        super(uNet19Att, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.attention = attention
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg19_bn(pretrained=pretrained).features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu, self.encoder[20], self.encoder[21], self.relu, self.encoder[23], self.encoder[24], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[27], self.encoder[28], self.relu, self.encoder[30], self.encoder[31], self.relu, self.encoder[33], self.encoder[34], self.relu, self.encoder[36], self.encoder[37], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[40], self.encoder[41], self.relu, self.encoder[43], self.encoder[44], self.relu, self.encoder[46], self.encoder[47], self.relu, self.encoder[49], self.encoder[50], self.relu)
        self.deblock1 = block.QuadripleUp(512,512,1024)
        self.deblock2 = block.QuadripleUp(512,256)
        self.deblock3 = block.DoubleUp(256,128)
        self.deblock4 = block.DoubleUp(128,64)

        self.down4 = block.SimpleConv(512, 16)
        self.down3 = block.SimpleConv(256, 16)
        self.down2 = block.SimpleConv(128, 16)
        self.down1 = block.SimpleConv(64, 16)
        
        self.down_out1 = block.SingleConv(16, self.n_classes)
        self.down_out2 = block.SingleConv(16, self.n_classes)
        self.down_out3 = block.SingleConv(16, self.n_classes)
        self.down_out4 = block.SingleConv(16, self.n_classes)
        
        if self.attention:
            self.mix1 = block.Attention(16, 16)
            self.mix2 = block.Attention(16, 16)
            self.mix3 = block.Attention(16, 16)
            self.mix4 = block.Attention(16, 16)
        
            self.mix_out1 = block.SingleConv(16, self.n_classes)
            self.mix_out2 = block.SingleConv(16, self.n_classes)
            self.mix_out3 = block.SingleConv(16, self.n_classes)
            self.mix_out4 = block.SingleConv(16, self.n_classes)
        
        self.out1 = block.SimpleConv(16*4, 64)
        self.out2 = block.SingleConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        dec4 = self.deblock1(x5, x4)
        dec3 = self.deblock2(dec4, x3)
        dec2 = self.deblock3(dec3, x2)
        dec1 = self.deblock4(dec2, x1)
        
        down1 = block.up_sample2d(self.down1(dec1), x)
        down2 = block.up_sample2d(self.down2(dec2), x)
        down3 = block.up_sample2d(self.down3(dec3), x)
        down4 = block.up_sample2d(self.down4(dec4), x)

        down_out1 = self.down_out1(down1)
        down_out2 = self.down_out2(down2)
        down_out3 = self.down_out3(down3)
        down_out4 = self.down_out4(down4)

        if self.attention:
            mix1, att1 = self.mix1(down1)
            mix2, att2 = self.mix2(down2)
            mix3, att3 = self.mix3(down3)
            mix4, att4 = self.mix4(down4)
            mix_out1 = self.mix_out1(mix1)
            mix_out2 = self.mix_out2(mix2)
            mix_out3 = self.mix_out3(mix3)
            mix_out4 = self.mix_out4(mix4)
            logits = self.out2(self.out1(torch.cat((mix1, mix2, mix3, mix4), dim=1)))
        else:
            logits = self.out2(self.out1(torch.cat((down1, down2, down3, down4), dim=1)))
        
        if self.training:
            if self.attention: 
                return logits, mix_out1, mix_out2, mix_out3, mix_out4, down_out1, down_out2, down_out3, down_out4
            else:
                return logits, down_out1, down_out2, down_out3, down_out4
        else:
            return logits