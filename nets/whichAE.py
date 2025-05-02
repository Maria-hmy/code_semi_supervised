#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nets import AE

def whichAE(net_id):
    
    if net_id == 1:
        net = AE.CAESimpleConv()
        
    elif net_id == 2:
        net = AE.CAEDoubleConv()
        
    elif net_id == 3:
        net = AE.OCAESimpleConv()
        
    return net