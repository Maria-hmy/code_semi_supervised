#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
from skimage import io
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, affine
from exams.exam_genkyst import exam_genkyst
from genkyst_MT.manage_genkyst_MT import extract_genkyst_slice_MT
from utils.utils import normalization_imgs, normalization_masks
 
class dataset_genkyst_MT(Dataset):

    def __init__(self, path:str, scheme:str, modality:str, vgg:bool=False, clfTask=False):
        self.path = path
        self.scheme = scheme
        self.modality = modality 
        self.anatomy = 'LK+RK'
        self.vgg = vgg
        self.ids = np.load(self.path+'imgs-id-'+self.scheme+'.npy')
        self.clfTask = clfTask
        if self.clfTask:
            self.mayos = np.load(self.path+'imgs-mayo-'+scheme+'.npy')
            self.convert_mayo()

    def __len__(self):
        return len(self.ids)
    
    def convert_mayo(self):
        self.mayos[np.where(self.mayos == '2')]  = '5'
        self.mayos[np.where(self.mayos == '?')]  = '2' # unknown mayo class
        self.mayos[np.where(self.mayos == '1A')] = '0'
        self.mayos[np.where(self.mayos == '1B')] = '1'
        self.mayos[np.where(self.mayos == '1C')] = '2'
        self.mayos[np.where(self.mayos == '1D')] = '3'
        self.mayos[np.where(self.mayos == '1E')] = '4'
        self.mayos = np.array(self.mayos, dtype=np.float32)

    def transform(self, img, mask_LK, mask_RK):
        (d,t,sc,sh) = transforms.RandomAffine.get_params(degrees=(-20,20), translate=(0.2,0.2), scale_ranges=(0.8,1.2), shears=(-20,20), img_size=img.shape)
        img = affine(to_pil_image(img), angle=d, translate=t, scale=sc, shear=sh)
        mask_LK = affine(to_pil_image(mask_LK), angle=d, translate=t, scale=sc, shear=sh)
        mask_RK = affine(to_pil_image(mask_RK), angle=d, translate=t, scale=sc, shear=sh)
        return (np.array(img), np.array(mask_LK), np.array(mask_RK))

    def __getitem__(self, idx:int):
        folder = self.path+self.scheme+'/'
        img = io.imread(folder+self.ids[idx]+'-src.png')
        mask_LK = io.imread(folder+self.ids[idx]+'-mask-LK.png')
        mask_RK = io.imread(folder+self.ids[idx]+'-mask-RK.png')
        if self.clfTask:
            mayo = self.mayos[idx]
        if self.scheme == 'train':
            img, mask_LK, mask_RK = self.transform(img, mask_LK, mask_RK)	
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.float32)
        img_[:,:,0] = img
        if self.vgg:
            img_[:,:,1], img_[:,:,2] = img, img
        mask_LK_ = np.zeros(shape=(mask_LK.shape[0],mask_LK.shape[1],1), dtype=np.uint8)
        mask_LK_[:,:,0] = mask_LK
        mask_RK_ = np.zeros(shape=(mask_RK.shape[0],mask_RK.shape[1],1), dtype=np.uint8)
        mask_RK_[:,:,0] = mask_RK
        img_ = normalization_imgs(img_)
        mask_LK_ = normalization_masks(mask_LK_)
        mask_RK_ = normalization_masks(mask_RK_)
        if self.clfTask:
            return (img_.swapaxes(2,0), mask_LK_.swapaxes(2,0), mask_RK_.swapaxes(2,0), mayo)
        else:
            return (img_.swapaxes(2,0), mask_LK_.swapaxes(2,0), mask_RK_.swapaxes(2,0))            
    
class tiny_dataset_genkyst_MT(Dataset):
    ''' one single examination for prediction purposes '''
    
    def __init__(self, id_, serie, size, modality, vgg:bool=False):
        self.id = id_
        self.serie = serie
        self.size = size
        self.modality = modality 
        self.anatomy = 'LK+RK'
        self.vgg = vgg
        self.exam = exam_genkyst(self.id, self.serie, self.modality, upload=True)
        self.exam.normalize(self.modality)
            
    def __len__(self):
        if self.modality == 'T2':
            return self.exam.T2.shape[1]
        elif self.modality == 'CT':
            return self.exam.CT.shape[2]

    def __getitem__(self, idx:int):
        img, mask_LK, mask_RK = extract_genkyst_slice_MT(self.exam, idx, self.modality, self.size)
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.float32)
        img_[:,:,0] = img
        if self.vgg:
            img_[:,:,1], img_[:,:,2] = img, img
        mask_LK_ = np.zeros(shape=(mask_LK.shape[0],mask_LK.shape[1],1), dtype=np.uint8)
        mask_LK_[:,:,0] = mask_LK
        mask_RK_ = np.zeros(shape=(mask_RK.shape[0],mask_RK.shape[1],1), dtype=np.uint8)
        mask_RK_[:,:,0] = mask_RK
        img_ = normalization_imgs(img_)
        mask_LK_ = normalization_masks(mask_LK_)
        mask_RK_ = normalization_masks(mask_RK_)
        return (img_.swapaxes(2,0), mask_LK_.swapaxes(2,0), mask_RK_.swapaxes(2,0))