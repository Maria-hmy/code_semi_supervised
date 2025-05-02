#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from manage.manage_genkyst import genksyt_split
from genkyst_MT.manage_genkyst_MT import create_genkyst_dataset_MT
from genkyst_MT.dataset_genkyst_MT import dataset_genkyst_MT
import argparse
import logging
import sys
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import distutils.dir_util
from utils.train_utils import launch_training_MT, dice_history
from nets.whichnet import whichnet

def train_genkyst(net_id, 
                  net,
                  modality,
                  trial,
                  output,
                  device,
                  vgg,
                  epochs,
                  batch,
                  lr,
                  size,
                  gamma):
    
    train_ids, val_ids, test_ids, train_series, val_series, test_series = genksyt_split(modality, trial)
    
    train_ids += test_ids
    train_series += test_series
    
    logging.info(f'''training ids: {train_ids}''')
    
    logging.info(f'''validation ids: {val_ids}''')
                 
    create_genkyst_dataset_MT(output, train_ids, train_series, 'train', modality, size)
    
    create_genkyst_dataset_MT(output, train_ids, train_series, 'val', modality, size) 

    train_dataset = dataset_genkyst_MT(output, 'train', modality, vgg)

    val_dataset = dataset_genkyst_MT(output, 'val', modality, vgg)    

    n_train, n_val = len(train_dataset), len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
  
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    
    logging.info(f'''starting training:
        anatomy:         {'LK+RK'}
        modality:        {modality}
        trial:           {trial}
        vgg:             {vgg}
        epochs:          {epochs}
        batch size:      {batch}
        learning rate:   {lr}
        training size:   {n_train}
        validation size: {n_val}
        size:            {size}
        gamma:           {gamma}
        device:          {device.type}''')
    
    optimizer = optim.Adam(net.parameters(), lr=1e-6*lr)
    
    criterion = nn.BCEWithLogitsLoss()
    
    return launch_training_MT(epochs, net, train_loader, val_loader, n_train, device, net_id, criterion, optimizer, output, gamma)

def get_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-o', '--output', type=str, default='./../../results/genkyst/', dest='output')
    
    parser.add_argument('-m', '--modality', type=str, default='T2', dest='modality')
    
    parser.add_argument('-e', '--epochs', type=int, default=500, dest='epochs')
    
    parser.add_argument('-b', '--batch', type=int, default=8, dest='batch')
    
    parser.add_argument('-t', '--trial', type=int, default=3, dest='trial')
    
    parser.add_argument('-l', '--learning', type=int, default=10, dest='lr')
    
    parser.add_argument('-n', '--network', type=int, default=27, dest='network')
    
    parser.add_argument('-s', '--size', type=int, default=256, dest='size')
    
    parser.add_argument('-g', '--gamma', type=int, default=5, dest='gamma')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info('using device: ' + str(device))
    
    n_classes, anatomy = 1, 'LK+RK'
    
    net, vgg = whichnet(args.network, n_classes, args.size)
    
    logging.info(f'network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    net.to(device=device)

    try:
        
        args.output += 'genkyst-'+args.modality+'-'+anatomy+'-t-'+str(args.trial)+'-n-'+str(args.network)+'-e-'+str(args.epochs)+'-b-'+str(args.batch)
        args.output += '-l-'+str(int(args.lr))+'-s-'+str(args.size)+'-g-'+str(args.gamma)+'/'

        distutils.dir_util.mkpath(args.output)
        
        train_dices, val_dices = train_genkyst(net_id = args.network, 
                                               net = net,
                                               modality = args.modality,
                                               trial = args.trial,
                                               output = args.output,
                                               device = device,
                                               vgg = vgg,
                                               epochs = args.epochs,
                                               batch = args.batch,
                                               lr = args.lr,
                                               size = args.size, 
                                               gamma = args.gamma)
        
        dice_history(args.epochs, train_dices, val_dices, args.output)

    except KeyboardInterrupt:
        logging.info('keyboard interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)