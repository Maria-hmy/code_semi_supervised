#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np
import torch
from utils.utils import prob2mask, get_array_affine_header, boundaries, print_ods, get2LargestConnectedAreas
from torch.utils.data import DataLoader
from skimage import io
import tqdm
import distutils.dir_util
from skimage import img_as_ubyte
import nibabel
from skimage.transform import resize, rotate
from utils.metric import assessment
from datasets.dataset_genkyst import tiny_dataset_genkyst
from manage.manage_genkyst import genksyt_split
from nets.whichnet import whichnet
    
def infer_genkyst(net,
                  net_id,
                  anatomy,
                  modality,
                  trial,
                  output,
                  device,
                  vgg,
                  size):

    _, _, test_ids, _, _, test_series = genksyt_split(modality, trial)
    
    for path in [output+'nii/', output+'bound/']:
        distutils.dir_util.mkpath(path)
    
    scores = np.zeros((4,len(test_ids),7), dtype=float)
    
    for index, id_ in enumerate(tqdm.tqdm(test_ids)):
        
        test_dataset = tiny_dataset_genkyst(id_, test_series[index], size, modality, anatomy, vgg)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        array, affine, header = get_array_affine_header(test_dataset, modality)

        with torch.no_grad():
            
            for idx, data in enumerate(test_loader):
                                
                id_slice = modality+'-%0*d'%(3,id_)+'-%0*d-'%(2,test_series[index])+anatomy+'-%0*d'%(3,idx+1)
                
                image, label = data
                image = image.to(device=device, dtype=torch.float32) 
                
                if net_id in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
                    net.training = False
                    
                prob = torch.sigmoid(net(image))
                mask = prob2mask(prob)
                
                mask = rotate(mask, -90, preserve_range=True)
                if modality == 'T2':
                    mask = resize(mask, output_shape=(test_dataset.exam.T2.shape[0],test_dataset.exam.T2.shape[2]), preserve_range=True)
                elif modality == 'CT':
                    mask = resize(mask, output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask[np.where(mask>0.95)] = 1
                mask[np.where(mask!=1)] = 0
                if modality == 'T2':
                    array[0:test_dataset.exam.T2.shape[0],idx,0:test_dataset.exam.T2.shape[2]] = mask[::-1,::]
                elif modality == 'CT':
                    array[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask[::-1,::]
                
        prediction_nopp = nibabel.Nifti1Image(array.astype(np.uint16), affine=affine, header=header) # no post-processing
        nibabel.save(prediction_nopp, output+'nii/%0*d'%(3,id_)+'-%0*d-prediction-nopp.nii.gz'%(2,test_series[index])) # no post-processing
        
        array1, array2 = get2LargestConnectedAreas(array, separate=True)
        
        min1X, min2X = np.min(np.where(array1>0)[0]), np.min(np.where(array2>0)[0])
        if min1X > min2X:
            array_RK = array1
            array_LK = array2
        else:
            array_RK = array2
            array_LK = array1            
        array = array_LK+array_RK
        
        prediction_LK = nibabel.Nifti1Image(array_LK.astype(np.uint16), affine=affine, header=header)
        prediction_RK = nibabel.Nifti1Image(array_RK.astype(np.uint16), affine=affine, header=header)
        del array_LK, array_RK, array1, array2
        prediction = nibabel.Nifti1Image(array.astype(np.uint16), affine=affine, header=header)
        
        nibabel.save(prediction_LK, output+'nii/%0*d'%(3,id_)+'-%0*d-prediction-LK.nii.gz'%(2,test_series[index]))
        nibabel.save(prediction_RK, output+'nii/%0*d'%(3,id_)+'-%0*d-prediction-RK.nii.gz'%(2,test_series[index]))
        nibabel.save(prediction, output+'nii/%0*d'%(3,id_)+'-%0*d-prediction.nii.gz'%(2,test_series[index]))
        
        ones = nibabel.Nifti1Image(np.ones(prediction.get_fdata().shape, dtype = np.uint16), affine = prediction.affine, header = prediction.header)
         
        factor_ = 2
        if modality == 'CT':
            factor_ = 8
        
        del array
        scores[0,index,:] = assessment(prediction_LK, test_dataset.exam.LK, ones, factor=factor_)
        del prediction_LK
        scores[1,index,:] = assessment(prediction_RK, test_dataset.exam.RK, ones, factor=factor_)
        del prediction_RK
        scores[2,index,:] = assessment(prediction_nopp, test_dataset.exam.BK, ones, factor=factor_) # no post-processing
        del prediction_nopp
        scores[3,index,:] = assessment(prediction, test_dataset.exam.BK, ones, factor=factor_)
        del ones
        
        logging.info(f'''dice for exam {id_} serie {test_series[index]}: {scores[3,index,0]}''')
        
        if modality == 'T2': 
            for y in range(test_dataset.exam.T2.shape[1]): # visu
                if len(np.where(test_dataset.exam.BK.get_fdata()[:,y,:]!=0)[0])>0: # if anatomy visible in groundtruth
                    src = rotate(test_dataset.exam.T2.get_fdata()[:,y,:], 90, preserve_range=True)[:,::-1]
                    gt_ = rotate(test_dataset.exam.BK.get_fdata()[:,y,:], 90, preserve_range=True)[:,::-1]
                    gt_[np.where(gt_>0.95)] = 1
                    gt_[np.where(gt_!=1)] = 0
                    gt_ = gt_.astype(np.uint8)
                    pd_ = rotate(prediction.get_fdata()[:,y,:], 90, preserve_range=True)[:,::-1]
                    pd_[np.where(pd_>0.95)] = 1
                    pd_[np.where(pd_!=1)] = 0
                    pd_ = pd_.astype(np.uint8)
                    id_slice = modality+'-%0*d'%(3,id_)+'-%0*d-'%(2,test_series[index])+anatomy+'-%0*d'%(3,y+1)
                    boundaries_ = boundaries(src, pd_, gt_)
                    io.imsave(output+'bound/'+id_slice+'-bound.png', img_as_ubyte(boundaries_))
        elif modality == 'CT': 
            for z in range(test_dataset.exam.CT.shape[2]): # visu
                if len(np.where(test_dataset.exam.BK.get_fdata()[:,:,z]!=0)[0])>0: # if anatomy visible in groundtruth
                    src = rotate(test_dataset.exam.CT.get_fdata()[:,:,z], 90, preserve_range=True)[:,::-1]
                    gt_ = rotate(test_dataset.exam.BK.get_fdata()[:,:,z], 90, preserve_range=True)[:,::-1]
                    gt_[np.where(gt_>0.95)] = 1
                    gt_[np.where(gt_!=1)] = 0
                    gt_ = gt_.astype(np.uint8)
                    pd_ = rotate(prediction.get_fdata()[:,:,z], 90, preserve_range=True)[:,::-1]
                    pd_[np.where(pd_>0.95)] = 1
                    pd_[np.where(pd_!=1)] = 0
                    pd_ = pd_.astype(np.uint8)
                    id_slice = modality+'-%0*d'%(3,id_)+'-%0*d-'%(2,test_series[index])+anatomy+'-%0*d'%(3,z+1)
                    boundaries_ = boundaries(src, pd_, gt_)
                    io.imsave(output+'bound/'+id_slice+'-bound.png', img_as_ubyte(boundaries_))            
        
        del prediction, test_dataset

    if len(np.unique(scores))>0:
        print_ods(scores[0,:,:], test_ids, test_series, output, 'overview-results-LK.ods')
        print_ods(scores[1,:,:], test_ids, test_series, output, 'overview-results-RK.ods')
        print_ods(scores[2,:,:], test_ids, test_series, output, 'overview-results-nopp.ods') # no post-processing
        print_ods(scores[3,:,:], test_ids, test_series, output, 'overview-results-BK.ods')

def get_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    
    parser.add_argument('-o', '--output', type=str, default='./../../results/genkyst/', dest='output')
    
    parser.add_argument('-m', '--modality', type=str, default='CT', dest='modality') # T2, CT, MR 
    
    parser.add_argument('-a', '--anatomy', type=str, default='BK', dest='anatomy') # LK, RK, BK
    
    parser.add_argument('-e', '--epochs', type=int, default=200, dest='epochs')
    
    parser.add_argument('-b', '--batch', type=int, default=16, dest='batch')
    
    parser.add_argument('-t', '--trial', type=int, default=1, dest='trial')
    
    parser.add_argument('-l', '--learning', type=int, default=10, dest='lr')
    
    parser.add_argument('-n', '--network', type=int, default=26, dest='network')
    
    parser.add_argument('-s', '--size', type=int, default=256, dest='size')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    args = get_args()

    args.output += 'genkyst-'+args.modality+'-'+args.anatomy+'-t-'+str(args.trial)+'-n-'+str(args.network)+'-e-'+str(args.epochs)+'-b-'+str(args.batch)
    args.output += '-l-'+str(int(args.lr))+'-s-'+str(args.size)+'/'
        
    args.model = args.output+'epoch.pth'
    
    args.output += 'test/'
    distutils.dir_util.mkpath(args.output)
    
    n_classes = 1
    net, vgg = whichnet(args.network, n_classes, args.size)

    logging.info("loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'using device {device}')
    
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("model loaded !")

    infer_genkyst(net = net,
                  net_id = args.network,
                  anatomy = args.anatomy,
                  modality = args.modality,
                  trial = args.trial,
                  output = args.output, 
                  device = device,
                  vgg = vgg,
                  size = args.size)