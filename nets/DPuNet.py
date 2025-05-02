#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
    
class DblePathuNet(nn.Module):
    
    def __init__(self, n_channels, n_classes):
        
        super(DblePathuNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.initblock = DoubleConv(self.n_channels,32)
        self.enblock1 = DoubleDown(32,64)
        # == 
        self.enblockA1 = DoubleDown(64,128)
        self.enblockB1 = DoubleConv(64,128)
        self.upFB1 = upFB(128, 2)
        self.downFB1 = downFB(128, 2)
        # == 
        self.enblockA2 = DoubleDown(128,256)
        self.enblockB2 = DoubleConv(128,256)
        self.upFB2 = upFB(256, 4)
        self.downFB2 = downFB(256, 4)
        # ==
        self.centerA = DoubleDown(256,256)
        self.centerB = DoubleConv(256,256)
        # == 
        self.deblockA1 = DoubleUp(256,256,512)
        self.deblockB1 = DoubleConvConcat(512,256)
        self.upFB3 = upFB(256, 4)
        self.downFB3 = downFB(256, 4)
        # ==
        self.deblockA2 = DoubleUp(256,128)
        self.deblockB2 = DoubleConvConcat(256+128,128)
        self.upFB4 = upFB(128, 2)
        self.downFB4 = downFB(128, 2)
        # ==
        self.deblockA3 = DoubleUp(128,64)
        self.deblockB3 = DoubleConvConcat(128+64,64)
        
        # ==
        self.simpleconv = SimpleConv(128,64)
        self.deblockA4 = DoubleUp(64,32)
        # ==        
        self.outc = OutConv(32, self.n_classes)

    def forward(self, x):
        
        x0 = self.initblock(x) # N ; 32
        x1 = self.enblock1(x0) # N/2 ; 64
        # == 
        x2_A = self.enblockA1(x1) # N/4 ; 128
        x2_B = self.enblockB1(x1) # N/2 ; 128
        x2_A = self.downFB1(x2_A, x2_B) # N/4 ; 128
        x2_B = self.upFB1(x2_A, x2_B) # N/2 ; 128   
        # ==
        x3_A = self.enblockA2(x2_A) # N/16 ; 256
        x3_B = self.enblockB2(x2_B) # N/2 ; 256
        x3_A = self.downFB2(x3_A, x3_B) # N/16 ; 256
        x3_B = self.upFB2(x3_A, x3_B) # N/2 ; 256     
        # == 
        x4_A = self.centerA(x3_A) # N/32 ; 256
        x4_B = self.centerB(x3_B) # N/2 ; 256
        # ==
        x_A = self.deblockA1(x4_A, x3_A) # N/16 ; 256
        x_B = self.deblockB1(x4_B, x3_B) # N/2 ; 256
        x_A = self.downFB3(x_A, x_B) # N/16 ; 256
        x_B = self.upFB3(x_A, x_B) # N/2 ; 256         
        # ==
        x_A = self.deblockA2(x_A, x2_A) # N/4 ; 128
        x_B = self.deblockB2(x_B, x2_B) # N/2 ; 128
        x_A = self.downFB4(x_A, x_B) # N/4 ; 128
        x_B = self.upFB4(x_A, x_B) # N/2 ; 128           
        # ==
        x_A = self.deblockA3(x_A, x1) # N/2 ; 64
        x_B = self.deblockB3(x_B, x1) # N/2 ; 64
        x = torch.cat([x_A, x_B], dim=1) # N/2 ; 128
        x = self.simpleconv(x) # N/2 ; 64
        x = self.deblockA4(x, x0) # N ; 32
        logits = self.outc(x)
        return logits
    
class DblePathuNet16(nn.Module):
    
    def __init__(self, n_classes, pretrained=True):
        
        super(DblePathuNet16, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg19_bn(pretrained=pretrained).features
        self.initblock = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu) # 64 filters
        self.enblock1 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu) # 128 filters
        
        # == 
        self.enblockA1 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu, self.encoder[20], self.encoder[21], self.relu) # 256 filters
        self.enblockB1 = TripleConv(128,256)
        self.upFB1 = upFB(256, 2)
        self.downFB1 = downFB(256, 2)
        # == 
        self.enblockA2 = nn.Sequential(self.pool, self.encoder[24], self.encoder[25], self.relu, self.encoder[27], self.encoder[28], self.relu, self.encoder[30], self.encoder[31], self.relu) # 512 filters
        self.enblockB2 = TripleConv(256,512)
        self.upFB2 = upFB(512, 4)
        self.downFB2 = downFB(512, 4)
        # ==
        self.centerA = nn.Sequential(self.pool, self.encoder[34], self.encoder[35], self.relu, self.encoder[37], self.encoder[38], self.relu, self.encoder[40], self.encoder[41], self.relu) # 512 filters
        self.centerB = TripleConv(512,512)
        # == 
        self.deblockA1 = TripleUp(512,512,1024)
        self.deblockB1 = TripleConvConcat(1024,512)
        self.upFB3 = upFB(512, 4)
        self.downFB3 = downFB(512, 4)
        # ==
        self.deblockA2 = TripleUp(512,256)
        self.deblockB2 = TripleConvConcat(512+256,256)
        self.upFB4 = upFB(256, 2)
        self.downFB4 = downFB(256, 2)
        # ==
        self.deblockA3 = DoubleUp(256,128)
        self.deblockB3 = DoubleConvConcat(256+128,128)
        # ==
        self.simpleconv = SimpleConv(256,128)
        self.deblockA4 = DoubleUp(128,64)
        # ==        
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        
        x0 = self.initblock(x) # N ; 64
        x1 = self.enblock1(x0) # N/2 ; 128
        # == 
        x2_A = self.enblockA1(x1) # N/4 ; 256
        x2_B = self.enblockB1(x1) # N/2 ; 256
        x2_A = self.downFB1(x2_A, x2_B) # N/4 ; 256
        x2_B = self.upFB1(x2_A, x2_B) # N/2 ; 256   
        # ==
        x3_A = self.enblockA2(x2_A) # N/16 ; 512
        x3_B = self.enblockB2(x2_B) # N/2 ; 512
        x3_A = self.downFB2(x3_A, x3_B) # N/16 ; 512
        x3_B = self.upFB2(x3_A, x3_B) # N/2 ; 512     
        # == 
        x4_A = self.centerA(x3_A) # N/32 ; 512
        x4_B = self.centerB(x3_B) # N/2 ; 512
        # ==
        x_A = self.deblockA1(x4_A, x3_A) # N/16 ; 512
        x_B = self.deblockB1(x4_B, x3_B) # N/2 ; 512
        x_A = self.downFB3(x_A, x_B) # N/16 ; 512
        x_B = self.upFB3(x_A, x_B) # N/2 ; 512         
        # ==
        x_A = self.deblockA2(x_A, x2_A) # N/4 ; 256
        x_B = self.deblockB2(x_B, x2_B) # N/2 ; 256
        x_A = self.downFB4(x_A, x_B) # N/4 ; 256
        x_B = self.upFB4(x_A, x_B) # N/2 ; 256           
        # ==
        x_A = self.deblockA3(x_A, x1) # N/2 ; 128
        x_B = self.deblockB3(x_B, x1) # N/2 ; 128
        x = torch.cat([x_A, x_B], dim=1) # N/2 ; 256
        x = self.simpleconv(x) # N/2 ; 128
        x = self.deblockA4(x, x0) # N ; 64
        logits = self.outc(x)
        return logits
    
class DblePathuNet19(nn.Module):
    
    def __init__(self, n_classes, pretrained=True):
        
        super(DblePathuNet19, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg19_bn(pretrained=pretrained).features
        self.initblock = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu) # 64 filters
        self.enblock1 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu) # 128 filters
        
        # == 
        self.enblockA1 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu, self.encoder[20], self.encoder[21], self.relu, self.encoder[23], self.encoder[24], self.relu) # 256 filters
        self.enblockB1 = QuadripleConv(128,256)
        self.upFB1 = upFB(256, 2)
        self.downFB1 = downFB(256, 2)
        # == 
        self.enblockA2 = nn.Sequential(self.pool, self.encoder[27], self.encoder[28], self.relu, self.encoder[30], self.encoder[31], self.relu, self.encoder[33], self.encoder[34], self.relu, self.encoder[36], self.encoder[37], self.relu) # 512 filters
        self.enblockB2 = QuadripleConv(256,512)
        self.upFB2 = upFB(512, 4)
        self.downFB2 = downFB(512, 4)
        # ==
        self.centerA = nn.Sequential(self.pool, self.encoder[40], self.encoder[41], self.relu, self.encoder[43], self.encoder[44], self.relu, self.encoder[46], self.encoder[47], self.relu, self.encoder[49], self.encoder[50], self.relu) # 512 filters
        self.centerB = QuadripleConv(512,512)
        # == 
        self.deblockA1 = QuadripleUp(512,512,1024)
        self.deblockB1 = QaudripleConvConcat(1024,512)
        self.upFB3 = upFB(512, 4)
        self.downFB3 = downFB(512, 4)
        # ==
        self.deblockA2 = QuadripleUp(512,256)
        self.deblockB2 = QaudripleConvConcat(512+256,256)
        self.upFB4 = upFB(256, 2)
        self.downFB4 = downFB(256, 2)
        # ==
        self.deblockA3 = DoubleUp(256,128)
        self.deblockB3 = DoubleConvConcat(256+128,128)
        # ==
        self.simpleconv = SimpleConv(256,128)
        self.deblockA4 = DoubleUp(128,64)
        # ==        
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        
        x0 = self.initblock(x) # N ; 64
        x1 = self.enblock1(x0) # N/2 ; 128
        # == 
        x2_A = self.enblockA1(x1) # N/4 ; 256
        x2_B = self.enblockB1(x1) # N/2 ; 256
        x2_A = self.downFB1(x2_A, x2_B) # N/4 ; 256
        x2_B = self.upFB1(x2_A, x2_B) # N/2 ; 256   
        # ==
        x3_A = self.enblockA2(x2_A) # N/16 ; 512
        x3_B = self.enblockB2(x2_B) # N/2 ; 512
        x3_A = self.downFB2(x3_A, x3_B) # N/16 ; 512
        x3_B = self.upFB2(x3_A, x3_B) # N/2 ; 512     
        # == 
        x4_A = self.centerA(x3_A) # N/32 ; 512
        x4_B = self.centerB(x3_B) # N/2 ; 512
        # ==
        x_A = self.deblockA1(x4_A, x3_A) # N/16 ; 512
        x_B = self.deblockB1(x4_B, x3_B) # N/2 ; 512
        x_A = self.downFB3(x_A, x_B) # N/16 ; 512
        x_B = self.upFB3(x_A, x_B) # N/2 ; 512         
        # ==
        x_A = self.deblockA2(x_A, x2_A) # N/4 ; 256
        x_B = self.deblockB2(x_B, x2_B) # N/2 ; 256
        x_A = self.downFB4(x_A, x_B) # N/4 ; 256
        x_B = self.upFB4(x_A, x_B) # N/2 ; 256           
        # ==
        x_A = self.deblockA3(x_A, x1) # N/2 ; 128
        x_B = self.deblockB3(x_B, x1) # N/2 ; 128
        x = torch.cat([x_A, x_B], dim=1) # N/2 ; 256
        x = self.simpleconv(x) # N/2 ; 128
        x = self.deblockA4(x, x0) # N ; 64
        logits = self.outc(x)
        return logits
    
class Interpolate(nn.Module):
    
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

class upFB(nn.Module):
    
    def __init__(self, channels, scale):
        super(upFB, self).__init__()
        self.interp = Interpolate(scale, 'bilinear')
        self.simple_conv = SimpleConv(2*channels, channels)
        
    def forward(self, x_A, x_B):
        return self.simple_conv(torch.cat([self.interp(x_A), x_B], dim=1))
    
class downFB(nn.Module):
    
    def __init__(self, channels, scale):
        super(downFB, self).__init__()
        self.maxpool = nn.MaxPool2d(scale, scale)
        self.simple_conv = SimpleConv(2*channels, channels)
        
    def forward(self, x_A, x_B):
        return self.simple_conv(torch.cat([x_A, self.maxpool(x_B)], dim=1))

class SimpleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.simple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))         
        
    def forward(self, x):
        return self.simple_conv(x)
    
class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))          
        
    def forward(self, x):
        return self.double_conv(x)
    
class TripleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.triple_conv(x)
    
class QuadripleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.quadriple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))          

    def forward(self, x):
        return self.quadriple_conv(x)
    
class DoubleDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.double_down(x)
  
class DoubleUp(nn.Module):

    def __init__(self, in_channels, out_channels, middle_channels=None):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        if middle_channels == None:
            middle_channels = in_channels
        self.conv = DoubleConv(middle_channels, out_channels)

    def forward(self, xA_d, xA_e):
        xA_d = self.up(xA_d)
        x = torch.cat([xA_d, xA_e], dim=1)
        return self.conv(x)
    
class TripleUp(nn.Module):

    def __init__(self, in_channels, out_channels, middle_channels=None):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)
        if middle_channels == None:
            middle_channels = in_channels
        self.conv = TripleConv(middle_channels, out_channels)

    def forward(self, xA_d, xA_e):
        xA_d = self.up(xA_d)
        x = torch.cat([xA_d, xA_e], dim=1)
        return self.conv(x)
    
class QuadripleUp(nn.Module):

    def __init__(self, in_channels, out_channels, middle_channels=None):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)
        if middle_channels == None:
            middle_channels = in_channels
        self.conv = QuadripleConv(middle_channels, out_channels)

    def forward(self, xA_d, xA_e):
        xA_d = self.up(xA_d)
        x = torch.cat([xA_d, xA_e], dim=1)
        return self.conv(x)
    
class DoubleConvConcat(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)       
        
    def forward(self, xB_d, xB_e):
        x = torch.cat([xB_d, xB_e], dim=1)
        return self.conv(x)
    
class TripleConvConcat(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = TripleConv(in_channels, out_channels)       
        
    def forward(self, xB_d, xB_e):
        x = torch.cat([xB_d, xB_e], dim=1)
        return self.conv(x)
    
class QaudripleConvConcat(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = QuadripleConv(in_channels, out_channels)       
        
    def forward(self, xB_d, xB_e):
        x = torch.cat([xB_d, xB_e], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)