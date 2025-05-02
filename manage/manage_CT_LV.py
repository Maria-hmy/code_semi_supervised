#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from exams.exam_halt import exam_halt
from exams.exam_berlin import exam_berlin
from manage.manage_halt import extract_halt_slice
import distutils.dir_util
from skimage import img_as_ubyte
from skimage import io
import tqdm
from utils import utils

def CT_LV_split(): 
    train_ids = ['007', '007', '017', '029', '029', '035', '041', '044']
    train_series = [1, 3, 1, 1, 2, 1, 1, 1]
    train_datasets = 8*['berlin']
      
    train_ids += ['010071', '010171', '019591', '019591', '031283', '031283', '038266', '048439', '068996', '300003', '300004', '300005']
    train_series += [1, 3, 1, 2, 2, 3, 1, 1, 1, 1, 1, 1]
    train_datasets += 12*['halt']    

    val_ids = ['030','043']
    val_series = [1, 1]
    val_datasets = ['berlin', 'berlin']

    val_ids += ['019289','208839']
    val_series += [1, 1]
    val_datasets += ['halt', 'halt']      
   
    return list(train_ids), list(val_ids), list(train_series), list(val_series), list(train_datasets), list(val_datasets)

def create_CT_LV_dataset(output, ids, series, datasets, scheme, size):
    list_id = []
    folder = output+scheme+'/'
    distutils.dir_util.mkpath(folder)
    dim = 2
    anatomy = 'LV'
    modality = 'CT'
    for idx, id_ in enumerate(tqdm.tqdm(ids)):
        if datasets[idx] == 'halt':
            exam = exam_halt(int(id_), series[idx], modality)
        elif datasets[idx] == 'berlin':
            exam = exam_berlin(int(id_), series[idx], modality)
        print(exam.id)
        exam.normalize(modality)  
        for xyz in range(exam.CT.shape[dim]): 
            img, mask = extract_halt_slice(exam, xyz, 'CT', size, anatomy)
            list_id.append(modality+'-%0*d'%(3,int(id_))+'-%0*d-'%(2,series[idx])+anatomy+'-%0*d'%(3,xyz+1))
            io.imsave(folder+list_id[-1]+'-src.png', img)
            io.imsave(folder+list_id[-1]+'-mask.png', mask, check_contrast=False)
            if len(np.unique(mask))>1:
                io.imsave(folder+list_id[-1]+'-bound.png', img_as_ubyte(utils.boundaries(img, None, mask)))
        del exam
    np.save(output+'imgs-id-'+scheme+'.npy', list_id)
    del list_id