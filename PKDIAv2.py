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
    
    net_id, n_classes, size, modality = 27, 1, 256, 'CT'
    
    net, vgg = whichnet(net_id, n_classes, size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'using device {device}')
    
    net.to(device=device)
    net.load_state_dict(torch.load('PKDIAv2-weights.pth', map_location=device))
    
    logging.info("model loaded !")
    
    test_dataset = tiny_dataset_genkyst_prod(args.patient, args.serie, size, args.input, args.output, modality, vgg)
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    array_LK, affine, header = get_array_affine_header(test_dataset, modality)
    array_RK, _, _ = get_array_affine_header(test_dataset, modality)
    array, _, _ = get_array_affine_header(test_dataset, modality)
    
    with torch.no_grad():
            
        for idx, data in enumerate(test_loader):
                                
            image = data
            image = image.to(device=device, dtype=torch.float32) 

            mask_LK = prob2mask(torch.sigmoid(net(image)[0]))
            mask_RK = prob2mask(torch.sigmoid(net(image)[1]))
                
            # == LK
            mask_LK = rotate(mask_LK, -90, preserve_range=True)
            mask_LK = resize(mask_LK, output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
            mask_LK[np.where(mask_LK>0.95)] = 1
            mask_LK[np.where(mask_LK!=1)] = 0
            array_LK[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_LK[::-1,::]
                    
            # == RK
            mask_RK = rotate(mask_RK, -90, preserve_range=True)
            mask_RK = resize(mask_RK, output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
            mask_RK[np.where(mask_RK>0.95)] = 1
            mask_RK[np.where(mask_RK!=1)] = 0
            array_RK[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_RK[::-1,::]
                    
        array = array_LK + array_RK
        array[np.where(array>0.)] = 1
        prediction_nopp = nibabel.Nifti1Image(array.astype(np.uint16), affine=affine, header=header) # no post-processing
        nibabel.save(prediction_nopp, args.output+'%0*d'%(6,args.patient)+'-%0*d-prediction-nopp.nii.gz'%(2,args.serie)) # no post-processing                   
            
        array_LK = getLargestConnectedArea(array_LK)
        array_RK = getLargestConnectedArea(array_RK)
        array = array_LK + array_RK
        array[np.where(array>0.)] = 1
        
        prediction_LK = nibabel.Nifti1Image(array_LK.astype(np.uint16), affine=affine, header=header)
        prediction_RK = nibabel.Nifti1Image(array_RK.astype(np.uint16), affine=affine, header=header)
        del array_LK, array_RK
        prediction = nibabel.Nifti1Image(array.astype(np.uint16), affine=affine, header=header)
        
        nibabel.save(prediction_LK, args.output+'%0*d'%(6,args.patient)+'-%0*d-prediction-LK.nii.gz'%(2,args.serie))
        nibabel.save(prediction_RK, args.output+'%0*d'%(6,args.patient)+'-%0*d-prediction-RK.nii.gz'%(2,args.serie))
        nibabel.save(prediction, args.output+'%0*d'%(6,args.patient)+'-%0*d-prediction.nii.gz'%(2,args.serie))