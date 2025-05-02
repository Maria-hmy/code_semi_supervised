#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np
import torch
from utils.utils import prob2mask, get_array_affine_header, boundaries, print_ods, getLargestConnectedArea
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
        
        test_dataset = tiny_dataset_genkyst(id_, test_series[index], size, modality, 'BK', vgg)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        array_LK, affine, header = get_array_affine_header(test_dataset, modality)
        array_RK, _, _ = get_array_affine_header(test_dataset, modality)
        array, _, _ = get_array_affine_header(test_dataset, modality)        

        with torch.no_grad():
            
            for idx, data in enumerate(test_loader):
                                
                image, _ = data
                image = image.to(device=device, dtype=torch.float32) 

                prob_LK = torch.sigmoid(net(image)[0])
                mask_LK = prob2mask(prob_LK)
                
                prob_RK = torch.sigmoid(net(image)[1])
                mask_RK = prob2mask(prob_RK)
                
                # == LK
                mask_LK = rotate(mask_LK, -90, preserve_range=True)
                if modality == 'T2':
                    mask_LK = resize(mask_LK, output_shape=(test_dataset.exam.T2.shape[0],test_dataset.exam.T2.shape[2]), preserve_range=True)
                elif modality == 'CT':
                    mask_LK = resize(mask_LK, output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_LK[np.where(mask_LK>0.95)] = 1
                mask_LK[np.where(mask_LK!=1)] = 0
                if modality == 'T2':
                    array_LK[0:test_dataset.exam.T2.shape[0],idx,0:test_dataset.exam.T2.shape[2]] = mask_LK[::-1,::]
                elif modality == 'CT':
                    array_LK[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_LK[::-1,::]
                    
                # == RK
                mask_RK = rotate(mask_RK, -90, preserve_range=True)
                if modality == 'T2':
                    mask_RK = resize(mask_RK, output_shape=(test_dataset.exam.T2.shape[0],test_dataset.exam.T2.shape[2]), preserve_range=True)
                elif modality == 'CT':
                    mask_RK = resize(mask_RK, output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_RK[np.where(mask_RK>0.95)] = 1
                mask_RK[np.where(mask_RK!=1)] = 0
                if modality == 'T2':
                    array_RK[0:test_dataset.exam.T2.shape[0],idx,0:test_dataset.exam.T2.shape[2]] = mask_RK[::-1,::]
                elif modality == 'CT':
                    array_RK[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_RK[::-1,::]
                    
        array = array_LK + array_RK
        array[np.where(array>0.)] = 1
        prediction_nopp = nibabel.Nifti1Image(array.astype(np.uint16), affine=affine, header=header) # no post-processing
        nibabel.save(prediction_nopp, output+'nii/%0*d'%(3,id_)+'-%0*d-prediction-nopp.nii.gz'%(2,test_series[index])) # no post-processing                   
            
        array_LK = getLargestConnectedArea(array_LK)
        array_RK = getLargestConnectedArea(array_RK)
        array = array_LK + array_RK
        array[np.where(array>0.)] = 1
        
        prediction_LK = nibabel.Nifti1Image(array_LK.astype(np.uint16), affine=affine, header=header)
        prediction_RK = nibabel.Nifti1Image(array_RK.astype(np.uint16), affine=affine, header=header)
        del array_LK, array_RK
        prediction = nibabel.Nifti1Image(array.astype(np.uint16), affine=affine, header=header)
        
        nibabel.save(prediction_LK, output+'nii/%0*d'%(3,id_)+'-%0*d-prediction-LK.nii.gz'%(2,test_series[index]))
        nibabel.save(prediction_RK, output+'nii/%0*d'%(3,id_)+'-%0*d-prediction-RK.nii.gz'%(2,test_series[index]))
        nibabel.save(prediction, output+'nii/%0*d'%(3,id_)+'-%0*d-prediction.nii.gz'%(2,test_series[index]))
        
        ones = nibabel.Nifti1Image(np.ones(prediction.get_fdata().shape, dtype = np.uint16), affine = prediction.affine, header = prediction.header)
         
        factor_ = 2 # 4
        if modality == 'CT':
            factor_ = 4 # 8
        

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
    
    parser.add_argument('-m', '--modality', type=str, default='T2', dest='modality')
    
    parser.add_argument('-a', '--anatomy', type=str, default='LK+RK', dest='anatomy')
    
    parser.add_argument('-e', '--epochs', type=int, default=6, dest='epochs') # 200
    
    parser.add_argument('-b', '--batch', type=int, default=2, dest='batch')
    
    parser.add_argument('-t', '--trial', type=int, default=0, dest='trial')
    
    parser.add_argument('-l', '--learning', type=int, default=10, dest='lr')
    
    parser.add_argument('-n', '--network', type=int, default=27, dest='network')
    
    parser.add_argument('-s', '--size', type=int, default=256, dest='size')
    
    parser.add_argument('-g', '--gamma', type=int, default=5, dest='gamma')
    
    parser.add_argument('-w', '--weights', type=str, default=None, dest='weights')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    args = get_args()

    args.output = args.output+'genkyst-'+args.modality+'-'+args.anatomy+'-t-'+str(args.trial)+'-n-'+str(args.network)+'-e-'+str(args.epochs)+'-b-'+str(args.batch)
    args.output += '-l-'+str(int(args.lr))+'-s-'+str(args.size)+'-g-'+str(args.gamma)
    if args.weights != None:
        args.output += '-w'
    args.output += '/'
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