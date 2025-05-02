#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from manage.manage_CT_LV import CT_LV_split, create_CT_LV_dataset
from datasets.dataset_genkyst import dataset_genkyst
import argparse
import logging
import sys
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import distutils.dir_util
from utils.train_utils import launch_training, dice_history
from nets.whichnet import whichnet

def train_CT_LV(net_id, 
                net,
                modality,
                output,
                device,
                vgg,
                epochs,
                batch,
                lr,
                size):
    
    train_ids, val_ids, train_series, val_series, train_datasets, val_datasets = CT_LV_split()
    
    logging.info(f'''training ids: {train_ids}''')
    
    logging.info(f'''validation ids: {val_ids}''')
                 
    create_CT_LV_dataset(output, train_ids, train_series, train_datasets, 'train', size)
    
    create_CT_LV_dataset(output, val_ids, val_series, val_datasets, 'val', size) 

    train_dataset = dataset_genkyst(output, 'train', modality, anatomy, vgg)

    val_dataset = dataset_genkyst(output, 'val', modality, anatomy, vgg)    

    n_train, n_val = len(train_dataset), len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
  
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    
    logging.info(f'''starting training:
        anatomy:         {'LK+RK'}
        modality:        {modality}
        vgg:             {vgg}
        epochs:          {epochs}
        batch size:      {batch}
        learning rate:   {lr}
        training size:   {n_train}
        validation size: {n_val}
        size:            {size}
        device:          {device.type}''')
    
    optimizer = optim.Adam(net.parameters(), lr=1e-6*lr)
    
    criterion = nn.BCEWithLogitsLoss()
    
    return launch_training(epochs, net, train_loader, val_loader, n_train, device, net_id, criterion, optimizer, output)

def get_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-o', '--output', type=str, default='./../../results/', dest='output')
    
    parser.add_argument('-m', '--modality', type=str, default='CT', dest='modality')
    
    parser.add_argument('-e', '--epochs', type=int, default=500, dest='epochs')
    
    parser.add_argument('-b', '--batch', type=int, default=16, dest='batch')
    
    parser.add_argument('-l', '--learning', type=int, default=10, dest='lr')
    
    parser.add_argument('-n', '--network', type=int, default=26, dest='network')
    
    parser.add_argument('-s', '--size', type=int, default=256, dest='size')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info('using device: ' + str(device))
    
    n_classes, anatomy = 1, 'LV'
    
    net, vgg = whichnet(args.network, n_classes, args.size)
    
    logging.info(f'network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    net.to(device=device)

    try:
        
        args.output += 'PKDIAv2'+args.modality+'-'+anatomy+'-n-'+str(args.network)+'-e-'+str(args.epochs)+'-b-'+str(args.batch)
        args.output += '-l-'+str(int(args.lr))+'-s-'+str(args.size)+'/'

        distutils.dir_util.mkpath(args.output)
        
        train_dices, val_dices = train_CT_LV(net_id = args.network, 
                                             net = net,
                                             modality = args.modality,
                                             output = args.output,
                                             device = device,
                                             vgg = vgg,
                                             epochs = args.epochs,
                                             batch = args.batch,
                                             lr = args.lr,
                                             size = args.size)
        
        dice_history(args.epochs, train_dices, val_dices, args.output)

    except KeyboardInterrupt:
        logging.info('keyboard interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)