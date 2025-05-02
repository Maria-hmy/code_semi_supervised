#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

# derived from https://github.com/jeya-maria-jose/KiU-Net-pytorch
    
class reskiUNet(nn.Module):
    
    def __init__(self, n_channels, n_classes):
        
        super(reskiUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.uNetEncoder_1  = uNetEncoder(self.n_channels, 16)
        self.kiNetEncoder_1 = kiNetEncoder(self.n_channels, 16)
        self.downCRFB_1     = downCRFB(16,4)
        self.upCRFB_1       = upCRFB(16,4)
        self.uNetEncoder_2  = uNetEncoder(16, 32)
        self.kiNetEncoder_2 = kiNetEncoder(16, 32)
        self.downCRFB_2     = downCRFB(32,16)
        self.upCRFB_2       = upCRFB(32,16)
        self.uNetEncoder_3  = uNetEncoder(32, 64)
        self.kiNetEncoder_3 = kiNetEncoder(32, 64)
        self.downCRFB_3     = downCRFB(64,64)
        self.upCRFB_3       = upCRFB(64,64)
        self.uNetDecoder_1  = kiNetEncoder(64,32)
        self.kiNetDecoder_1 = uNetEncoder(64,32)
        self.downCRFB_4     = downCRFB(32,16)
        self.upCRFB_4       = upCRFB(32,16)
        self.uNetDecoder_2  = kiNetEncoder(32,16)
        self.kiNetDecoder_2 = uNetEncoder(32,16)
        self.downCRFB_5     = downCRFB(16,4)
        self.upCRFB_5       = upCRFB(16,4)
        self.uNetDecoder_3  = kiNetEncoder(16,8)
        self.kiNetDecoder_3 = uNetEncoder(16,8)
        self.outc = OutConv(8, self.n_classes)
    
    def forward(self, x):

        # encoder
        out, out1 = self.uNetEncoder_1(x), self.kiNetEncoder_1(x)
        tmp = out
        out, out1 = torch.add(out, self.downCRFB_1(out1)), torch.add(out1, self.upCRFB_1(tmp))
        u1, o1 = out, out1
        out, out1 =  self.uNetEncoder_2(out), self.kiNetEncoder_2(out1)
        tmp = out
        out, out1 = torch.add(out, self.downCRFB_2(out1)), torch.add(out1, self.upCRFB_2(tmp))
        u2, o2 = out, out1
        out, out1 = self.uNetEncoder_3(out), self.kiNetEncoder_3(out1)
        tmp = out
        out, out1 = torch.add(out, self.downCRFB_3(out1)), torch.add(out1, self.upCRFB_3(tmp))
        
        # decoder
        out, out1 = self.uNetDecoder_1(out), self.kiNetDecoder_1(out1)
        tmp = out
        out, out1 = torch.add(out, self.downCRFB_4(out1)), torch.add(out1, self.upCRFB_4(tmp))
        out, out1 = torch.add(out,u2), torch.add(out1,o2)
        out, out1 = self.uNetDecoder_2(out), self.kiNetDecoder_2(out1)
        tmp = out
        out, out1 = torch.add(out, self.downCRFB_5(out1)), torch.add(out1, self.upCRFB_5(tmp))       
        out, out1 = torch.add(out,u1), torch.add(out1,o1)
        out, out1 = self.uNetDecoder_3(out), self.kiNetDecoder_3(out1) 
        out = torch.add(out,out1)

        logits = self.outc(out)
        return logits
    
class uNetEncoder(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.uNetEncoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))          
        
    def forward(self, x):
        return self.uNetEncoder(x)    
    
class kiNetEncoder(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.kiNetEncoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2), # ConvTranspose2d instead of Interpolate
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.kiNetEncoder(x)
    
class downCRFB(nn.Module):
    
    def __init__(self, channels, scale):
        super().__init__()
        self.downCRFB = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(scale, scale))          
        
    def forward(self, x):
        return self.downCRFB(x)
    
class upCRFB(nn.Module):
    
    def __init__(self, channels, scale):
        super().__init__()
        self.upCRFB = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)) # ConvTranspose2d instead of Interpolate         
        
    def forward(self, x):
        return self.upCRFB(x)
    
class OutConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)