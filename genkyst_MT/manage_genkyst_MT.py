#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from exams.exam_genkyst import exam_genkyst
import distutils.dir_util
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.transform import resize, rotate
from skimage import io
import tqdm

def create_genkyst_dataset_MT(output, ids, series, scheme, modality, size, clfTask=False):
    ''' create genkyst dataset with scheme={train,val,test}, modality={T2,MR,CT} '''
    
    list_id = []
    if clfTask:
        list_mayo = []
    folder = output+scheme+'/'
    distutils.dir_util.mkpath(folder)
    anatomy = 'LK+RK'
    dim = 2 if modality == 'CT' else 1
    
    for idx, id_ in enumerate(tqdm.tqdm(ids)):
        exam = exam_genkyst(id_,series[idx],modality)
        print(exam.id)
        exam.normalize(modality)
        ref = exam.CT if modality == 'CT' else exam.T2
        for xyz in range(ref.shape[dim]): 
            img, mask_LK, mask_RK = extract_genkyst_slice_MT(exam, xyz, modality, size)
            list_id.append(modality+'-%0*d'%(3,id_)+'-%0*d-'%(2,series[idx])+anatomy+'-%0*d'%(3,xyz+1))
            if clfTask:
                list_mayo.append(exam.mayo) # 1A, 1B, 1C, 1D, 1E, 2
            io.imsave(folder+list_id[-1]+'-src.png', img)
            io.imsave(folder+list_id[-1]+'-mask-LK.png', mask_LK, check_contrast=False)
            io.imsave(folder+list_id[-1]+'-mask-RK.png', mask_RK, check_contrast=False)
        del exam
    np.save(output+'imgs-id-'+scheme+'.npy', list_id)
    if clfTask:
        np.save(output+'imgs-mayo-'+scheme+'.npy', list_mayo)
    del list_id
    
def extract_genkyst_slice_MT(exam, xyz, modality, size):
    
    if modality == 'T2':
        img = rotate(resize(np.squeeze(exam.T2.get_fdata()[:,xyz,:])[::-1,:], output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        mask_LK = rotate(resize(exam.LK.get_fdata()[:,xyz,:][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        mask_RK = rotate(resize(exam.RK.get_fdata()[:,xyz,:][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    elif modality == 'CT':
        img = rotate(resize(np.squeeze(exam.CT.get_fdata()[:,:,xyz])[::-1,:], output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        mask_LK = rotate(resize(exam.LK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        mask_RK = rotate(resize(exam.RK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    
    min_greyscale, max_greyscale = np.percentile(img,(1,99)) 
    img = rescale_intensity(img, in_range=(min_greyscale,max_greyscale), out_range=(0,1))
    mask_LK[np.where(mask_LK>0)] = 255
    mask_RK[np.where(mask_RK>0)] = 255
    
    return img_as_ubyte(img), mask_LK.astype(np.uint8), mask_RK.astype(np.uint8)