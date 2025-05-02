#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np
import torch
from utils.utils import prob2mask, get_array_affine_header, getLargestConnectedArea
from torch.utils.data import DataLoader
import nibabel
from skimage.transform import resize, rotate
from datasets.dataset_genkyst import tiny_dataset_genkyst_prod
from nets.whichnet import whichnet

def get_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)   
    
    parser.add_argument('-i', '--input', type=str, default='./', dest='input')
    
    parser.add_argument('-o', '--output', type=str, default='./', dest='output')
    
    parser.add_argument('-p', '--patient', type=int, dest='patient')
    
    parser.add_argument('-s', '--serie', type=int, dest='serie')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    args = get_args()
    
    net_id, n_classes, size, modality = 26, 1, 256, 'CT'
    
    net, vgg = whichnet(net_id, n_classes, size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'using device {device}')
    
    net.to(device=device)
    net.load_state_dict(torch.load('PKDIAv2-CT-LV-weights.pth', map_location=device))
    
    logging.info("model loaded !")
    
    test_dataset = tiny_dataset_genkyst_prod(args.patient, args.serie, size, args.input, args.output, modality, vgg)
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    array_LV, affine, header = get_array_affine_header(test_dataset, modality)
    
    with torch.no_grad():
            
        for idx, data in enumerate(test_loader):
                                
            image = data
            image = image.to(device=device, dtype=torch.float32) 

            mask_LV = prob2mask(torch.sigmoid(net(image)))
            mask_LV = rotate(mask_LV, -90, preserve_range=True)
            mask_LV = resize(mask_LV, output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
            mask_LV[np.where(mask_LV>0.95)] = 1
            mask_LV[np.where(mask_LV!=1)] = 0
            array_LV[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_LV[::-1,::]

        prediction_nopp = nibabel.Nifti1Image(array_LV.astype(np.uint16), affine=affine, header=header) # no post-processing
        nibabel.save(prediction_nopp, args.output+'%0*d'%(6,args.patient)+'-%0*d-prediction-nopp.nii.gz'%(2,args.serie)) # no post-processing                   
            
        array_LV = getLargestConnectedArea(array_LV)
        prediction_LV = nibabel.Nifti1Image(array_LV.astype(np.uint16), affine=affine, header=header)
        nibabel.save(prediction_LV, args.output+'%0*d'%(6,args.patient)+'-%0*d-prediction-LV.nii.gz'%(2,args.serie))