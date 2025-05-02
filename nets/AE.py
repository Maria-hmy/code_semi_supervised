#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from nets import block
import torch

class CAESimpleConv(nn.Module):

    def __init__(self):
        super(CAESimpleConv, self).__init__()
        self.n_channels = 1
        self.n_classes  = 1

        self.enblock1   = block.SimpleConv(1, 8)
        self.enblock2   = block.SimpleDown(8,16)
        self.center     = block.SimpleDown(16,32)
        self.deblock1   = block.SimpleUpAE(32,16)
        self.deblock2   = block.SimpleUpAE(16,8)
        self.outc       = block.OutConv(8, 1)

    def forward(self, x):
        x = self.enblock1(x)
        x = self.enblock2(x)
        latent = self.center(x)
        z = self.deblock1(latent)
        z = self.deblock2(z)
        z = self.outc(z)
        return z, latent
    
class OCAESimpleConv(nn.Module):

    def __init__(self):
        super(OCAESimpleConv, self).__init__()
        self.n_channels = 1
        self.n_classes  = 1

        self.enblock1     = block.SimpleConv(1,8)
        self.enblock1_U   = block.SimpleDown(8,16)
        self.enblock2_U   = block.SimpleDown(16,32)
        self.enblock1_O   = block.SimpleUpAE(8,16)
        self.enblock2_O   = block.SimpleUpAE(16,32)
        self.collapse     = nn.MaxPool2d(16,16)
        self.deblock1     = block.SimpleUpAE(32+32,16)
        self.deblock2     = block.SimpleUpAE(16,8)
        self.outc         = block.OutConv(8,1)
        
    def forward(self, x):
        x = self.enblock1(x)
        x_U = self.enblock1_U(x)
        x_U = self.enblock2_U(x_U)
        x_O = self.enblock1_O(x)
        x_O = self.enblock2_O(x_O)
        x_O = self.collapse(x_O)
        latent = torch.cat([x_U, x_O], dim=1)
        z = self.deblock1(latent)
        z = self.deblock2(z)
        z = self.outc(z)
        return z, latent

class CAEDoubleConv(nn.Module):

    def __init__(self, n_classes):
        super(CAEDoubleConv, self).__init__()
        self.n_channels, self.n_classes = 1, n_classes
        self.n_classes  = 1

        self.enblock1   = block.DoubleConv(1, 32)
        self.enblock2   = block.DoubleDown(32,64)
        self.center     = block.DoubleDown(64,128)
        self.deblock1   = block.DoubleUpAE(128,64)
        self.deblock2   = block.DoubleUpAE(64,32)
        self.outc       = block.OutConv(32, 1)

    def forward(self, x):
        x = self.enblock1(x)
        x = self.enblock2(x)
        latent = self.center(x)
        z = self.deblock1(latent)
        z = self.deblock2(z)
        z = self.outc(z)
        return z, latent