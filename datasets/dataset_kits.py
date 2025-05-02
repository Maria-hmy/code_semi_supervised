#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
from skimage import io
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, affine
from exams.exam_halt import exam_halt
from exams.exam_halt_prod import exam_halt_prod
from manage.manage_halt import extract_halt_slice, extract_halt_slice_prod
from utils.utils import normalization_imgs, normalization_masks
 
class dataset_kits(Dataset):

    def __init__(self, path:str, scheme:str, anatomy:str, vgg:bool=False):
        self.path = path
        self.scheme = scheme
        self.anatomy = anatomy
        self.vgg = vgg
        self.ids = np.load(self.path+'imgs-id-'+self.scheme+'.npy')

    def __len__(self):
        return len(self.ids)

    def transform(self, img, mask):
        (d,t,sc,sh) = transforms.RandomAffine.get_params(degrees=(-20,20), translate=(0.2,0.2), scale_ranges=(0.8,1.2), shears=(-20,20), img_size=img.shape)
        img = affine(to_pil_image(img), angle=d, translate=t, scale=sc, shear=sh)
        mask = affine(to_pil_image(mask), angle=d, translate=t, scale=sc, shear=sh)
        return (np.array(img), np.array(mask))

    def __getitem__(self, idx:int):
        folder = self.path+self.scheme+'/'
        img = io.imread(folder+self.ids[idx]+'-src.png')
        mask = io.imread(folder+self.ids[idx]+'-mask.png')
        if self.scheme == 'train':
            img, mask = self.transform(img, mask)	
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.float32)
        img_[:,:,0] = img
        if self.vgg:
            img_[:,:,1], img_[:,:,2] = img, img
        mask_ = np.zeros(shape=(mask.shape[0],mask.shape[1],1), dtype=np.uint8)
        mask_[:,:,0] = mask
        img_ = normalization_imgs(img_)
        mask_ = normalization_masks(mask_)
        return (img_.swapaxes(2,0), mask_.swapaxes(2,0))
    
class dataset_kits_MT(Dataset):

    def __init__(self, path:str, scheme:str, vgg:bool=False):
        self.path = path
        self.scheme = scheme
        self.anatomy = 'LK+RK'
        self.vgg = vgg
        self.ids = np.load(self.path+'imgs-id-'+self.scheme+'.npy')

    def __len__(self):
        return len(self.ids)

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
        return (img_.swapaxes(2,0), mask_LK_.swapaxes(2,0), mask_RK_.swapaxes(2,0)) 