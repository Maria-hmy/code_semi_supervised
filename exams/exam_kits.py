#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel
import logging
from utils.utils import normalization_imgs, get2LargestConnectedAreas
import numpy as np

class exam_kits:

    def __init__(self, id_, upload=True):
        self.root = './../../data/kits2023/'
        self.folder = self.root+'dataset/'
        self.id = '%0*d'%(5,id_)
        if upload:
            self.exam_upload()
            self.print_info()

    def exam_upload(self):
        self.CT = nibabel.as_closest_canonical(nibabel.load(self.folder+'case_'+self.id+'/imaging.nii.gz'))
        self.BK = nibabel.as_closest_canonical(nibabel.load(self.folder+'case_'+self.id+'/segmentation.nii.gz'))
        self.BK.dataobj[np.where(self.BK.dataobj>0)] = 1
                                   
        array1, array2 = get2LargestConnectedAreas(self.BK.dataobj, separate=True)
        min1X, min2X = np.min(np.where(array1>0)[0]), np.min(np.where(array2>0)[0])
        if min1X > min2X:
            array_RK = array1
            array_LK = array2
        else:
            array_RK = array2
            array_LK = array1  
        self.LK = nibabel.Nifti1Image(array_LK.astype(np.uint16), affine=self.BK.affine, header=self.BK.header)
        self.RK = nibabel.Nifti1Image(array_RK.astype(np.uint16), affine=self.BK.affine, header=self.BK.header)
        del array_LK, array_RK, array1, array2
        
    def normalize(self):
        self.CT.get_fdata()[:,:,:] = normalization_imgs(self.CT.get_fdata())[:,:,:]
            
    def print_info(self):
        logging.basicConfig(level=logging.INFO, format='\n %(levelname)s: %(message)s')
        logging.info(f'''exam {self.id} uploaded:
        ''')