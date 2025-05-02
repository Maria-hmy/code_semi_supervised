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
from utils import utils

def create_genkyst_dataset(output, ids, series, scheme, modality, size, anatomy):
    list_id = []
    folder = output+scheme+'/'
    distutils.dir_util.mkpath(folder)
    dim = 2 if modality == 'CT' else 1
    for idx, id_ in enumerate(tqdm.tqdm(ids)):
        exam = exam_genkyst(id_,series[idx],modality)
        print(exam.id)
        exam.normalize(modality)  
        ref = exam.CT if modality == 'CT' else exam.T2
        for xyz in range(ref.shape[dim]): 
            img, mask = extract_genkyst_slice(exam, xyz, modality, size, anatomy)
            list_id.append(modality+'-%0*d'%(3,id_)+'-%0*d-'%(2,series[idx])+anatomy+'-%0*d'%(3,xyz+1))
            io.imsave(folder+list_id[-1]+'-src.png', img)
            io.imsave(folder+list_id[-1]+'-mask.png', mask, check_contrast=False)
            # if len(np.unique(mask))>1:
            #     io.imsave(folder+list_id[-1]+'-bound.png', img_as_ubyte(utils.boundaries(img, None, mask)))
        del exam
    np.save(output+'imgs-id-'+scheme+'.npy', list_id)
    del list_id
    
def create_genkyst_dataset_MT(output, ids, series, scheme, modality, size):
    list_id = []
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
            io.imsave(folder+list_id[-1]+'-src.png', img)
            io.imsave(folder+list_id[-1]+'-mask-LK.png', mask_LK, check_contrast=False)
            io.imsave(folder+list_id[-1]+'-mask-RK.png', mask_RK, check_contrast=False)
            # if len(np.unique(mask_LK))>1:
            #     io.imsave(folder+list_id[-1]+'-bound-LK.png', img_as_ubyte(utils.boundaries(img, None, mask_LK)))
            # if len(np.unique(mask_RK))>1:
            #     io.imsave(folder+list_id[-1]+'-bound-RK.png', img_as_ubyte(utils.boundaries(img, None, mask_RK)))
        del exam
    np.save(output+'imgs-id-'+scheme+'.npy', list_id)
    del list_id

def create_genkyst_dataset_3O(output, ids, series, scheme, modality, size):
    list_id = []
    folder = output+scheme+'/'
    distutils.dir_util.mkpath(folder)
    anatomy = '30'
    dim = 2 if modality == 'CT' else 1
    for idx, id_ in enumerate(tqdm.tqdm(ids)):
        exam = exam_genkyst(id_,series[idx],modality)
        print(exam.id)
        exam.normalize(modality)
        ref = exam.CT if modality == 'CT' else exam.T2
        for xyz in range(ref.shape[dim]): 
            img, mask_LK, mask_RK, mask_LV = extract_genkyst_slice_3O(exam, xyz, modality, size)
            list_id.append(modality+'-%0*d'%(3,id_)+'-%0*d-'%(2,series[idx])+anatomy+'-%0*d'%(3,xyz+1))
            io.imsave(folder+list_id[-1]+'-src.png', img)
            io.imsave(folder+list_id[-1]+'-mask-LK.png', mask_LK, check_contrast=False)
            io.imsave(folder+list_id[-1]+'-mask-RK.png', mask_RK, check_contrast=False)
            if exam.liv_annot:
                io.imsave(folder+list_id[-1]+'-mask-LV.png', mask_LV, check_contrast=False)
        del exam
    np.save(output+'imgs-id-'+scheme+'.npy', list_id)
    del list_id


################""
    
from skimage.transform import resize, rotate
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte
import numpy as np

def extract_genkyst_slice(exam, xyz, modality, size, anatomy):
    # === Chargement de l'image ===
    if modality == 'T2':
        img = exam.T2.get_fdata()[:, xyz, :][::-1, :]
    elif modality == 'CT':
        img = exam.CT.get_fdata()[:, :, xyz][::-1, :]
    else:
        raise ValueError(f"Modality '{modality}' non supportée")

    img = rotate(resize(img, output_shape=(size, size), preserve_range=True), 90, preserve_range=True)

    # === Initialisation du masque ===
    mask = np.zeros_like(img)

    # === Génération du masque ===
    def get_mask_from_field(field):
        raw = field.get_fdata()
        if modality == 'T2':
            m = raw[:, xyz, :][::-1, :] * 255
        else:  # CT
            m = raw[:, :, xyz][::-1, :] * 255
        return rotate(resize(m, output_shape=(size, size), preserve_range=True), 90, preserve_range=True)

    if anatomy in ['LK', 'RK', 'BK', 'LV']:
        mask = get_mask_from_field(getattr(exam, anatomy))
    elif anatomy == 'LK+RK':
        mask_lk = get_mask_from_field(exam.LK)
        mask_rk = get_mask_from_field(exam.RK)
        mask = np.maximum(mask_lk, mask_rk)
    else:
        raise ValueError(f"Anatomie '{anatomy}' non supportée")

    # === Normalisation de l'image ===
    min_greyscale, max_greyscale = np.percentile(img, (1, 99))
    img = rescale_intensity(img, in_range=(min_greyscale, max_greyscale), out_range=(0, 1))

    # === Seuillage du masque ===
    mask[mask > 0] = 255

    return img_as_ubyte(img), mask.astype(np.uint8)


#################

from skimage.transform import resize, rotate
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte
import numpy as np

def test_normalization(exam, xyz, size, modality='T2'):
    img = np.squeeze(exam.T2.get_fdata()[:, xyz, :])[::-1, :]
    img = rotate(resize(img, output_shape=(size, size), preserve_range=True), 90, preserve_range=True)

    # Normalisation identique au training
    min_grey, max_grey = np.percentile(img, (1, 99))
    img = rescale_intensity(img, in_range=(min_grey, max_grey), out_range=(0, 1))
    img = img_as_ubyte(img)  # convertit en [0, 255]
    img = img.astype(np.float32) / 255.0  # convertit en [0.0, 1.0]

    return img






################

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

def extract_genkyst_slice_3O(exam, xyz, modality, size):
    if modality == 'T2':
        img = rotate(resize(np.squeeze(exam.T2.get_fdata()[:,xyz,:])[::-1,:], output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        if exam.kid_annot:
            mask_LK = rotate(resize(exam.LK.get_fdata()[:,xyz,:][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
            mask_RK = rotate(resize(exam.RK.get_fdata()[:,xyz,:][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        else:
            mask_LK, mask_RK = None, None
        if exam.liv_annot:
            mask_LV = rotate(resize(exam.LV.get_fdata()[:,xyz,:][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        else:
            mask_LV = None
    elif modality == 'CT':
        img = rotate(resize(np.squeeze(exam.CT.get_fdata()[:,:,xyz])[::-1,:], output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        if exam.kid_annot:
            mask_LK = rotate(resize(exam.LK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
            mask_RK = rotate(resize(exam.RK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        else:
            mask_LK, mask_RK = None, None
        if exam.liv_annot:
            mask_LV = rotate(resize(exam.LV.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        else:
            mask_LV = None
    min_greyscale, max_greyscale = np.percentile(img,(1,99)) 
    img = rescale_intensity(img, in_range=(min_greyscale,max_greyscale), out_range=(0,1))
    if exam.kid_annot:
        mask_LK[np.where(mask_LK>0)] = 255
        mask_RK[np.where(mask_RK>0)] = 255
        mask_LK, mask_RK = mask_LK.astype(np.uint8), mask_RK.astype(np.uint8)
    if exam.liv_annot:
        mask_LV[np.where(mask_LV>0)] = 255
        mask_LV = mask_LV.astype(np.uint8)
    return img_as_ubyte(img), mask_LK, mask_RK, mask_LV

def extract_genkyst_slice_prod(exam, xyz, size, modality='T2'):
    if modality == 'T2':
        img = rotate(resize(np.squeeze(exam.T2.get_fdata()[:,xyz,:])[::-1,:], output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    elif modality == 'CT':
        img = rotate(resize(np.squeeze(exam.CT.get_fdata()[:,:,xyz])[::-1,:], output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    min_greyscale, max_greyscale = np.percentile(img,(1,99)) 
    img = rescale_intensity(img, in_range=(min_greyscale,max_greyscale), out_range=(0,1))
    return img_as_ubyte(img)

def genksyt_split(modality, trial): 
    if modality == 'T2':
        if trial == 0: # 186 18 50
            train_ids = [10024, 10024, 88648, 10089, 18190, 18896, 12990, 10032, 10032, 90006, 10168, 10168, 19523, 19523, 18132, 19525, 12981, 10882, 18900, 12664, 10005, 11902, 12296, 10854, 10854, 19189, 10856, 21193, 21193, 68549, 68549, 12727, 10886, 10886, 10877, 10840, 10840, 10006, 10186, 12450, 10885, 19606, 12743, 10135, 10862, 10862, 128249, 128249, 18511, 18511, 10839, 10839, 18899, 12557, 10871, 10871, 19595, 12405, 19288, 12535, 48882, 19524, 12908, 10157, 10157, 60034, 10124, 128247, 12582, 19610, 19610, 10890, 10890, 19605, 19605, 52400, 52326, 18898, 18898, 10073, 10073, 89221, 10870, 120007, 10848, 70007, 49270, 12628, 10173, 52608, 19231, 120015, 18314, 10873, 60005, 18133, 18133, 60952, 12558, 12531, 12438, 10007, 10007, 18315, 19604, 12747, 10154, 12984, 10170, 10136, 10879, 10879, 10879, 10087, 10842, 10842, 10156, 17823, 17823, 10065, 10065, 10065, 19356, 19356, 19356, 12402, 12983, 19608, 12583, 18309, 209135, 19286, 19286, 19286, 19286, 10174, 10066, 10066, 10151, 100018, 10896, 12447, 10044, 10044, 10176, 10176, 10057, 10057, 12335, 18188, 18188, 19233, 19233, 12446, 18189, 10866, 10866, 10036, 10036, 18091, 18091, 40012, 19594, 91014, 18920, 10867, 62046, 19236, 19236, 12331, 10171, 10171, 11903, 100015, 100015, 300007, 19287, 13041, 12298, 10134, 10178, 10178, 10887, 91343, 91343, 12905]
            train_series = [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 1, 1, 2, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
            val_ids = [10895, 10103, 10103, 10184, 10042, 60033, 52852, 11905, 19521, 10131, 52655, 19237, 12297, 12297, 18306, 52399, 12630, 12909]
            val_series = [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
            test_ids = [10026, 10037, 10041, 10045, 10045, 10064, 10076, 10076, 10082, 10094, 10179, 10179, 10187, 10187, 10187, 10843, 10843, 10851, 10855, 10869, 10869, 10869, 10880, 10880, 10892, 10897, 10897, 10898, 10898, 12483, 12532, 12560, 12627, 12665, 12728, 12846, 12901, 13044, 18310, 19119, 19188, 19285, 19359, 19401, 19454, 58284, 60023, 90027, 108192, 120985]
            test_series = [1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 3, 1, 2, 1, 1, 1, 2, 3, 1, 2, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif trial == 1: # 186 22 46
            train_ids = [10076, 10076, 12743, 18190, 58284, 19233, 19233, 40012, 108192, 91014, 18189, 10135, 12532, 10897, 10897, 10871, 10871, 10136, 10057, 10057, 10005, 10187, 10187, 10187, 18511, 18511, 10886, 10886, 10131, 10036, 10036, 10176, 10176, 10843, 10843, 10895, 12630, 10066, 10066, 62046, 19401, 12296, 88648, 17823, 17823, 10885, 10851, 89221, 10892, 10186, 19359, 60033, 12909, 12483, 19356, 19356, 19356, 19286, 19286, 19286, 19286, 10171, 10171, 10856, 11902, 10170, 10082, 12557, 10044, 10044, 19595, 10124, 10154, 209135, 18898, 18898, 10867, 10042, 18314, 300007, 10041, 10898, 10898, 12331, 10869, 10869, 10869, 10134, 10887, 52326, 19188, 12665, 18315, 52852, 12583, 18132, 19285, 12446, 12560, 10873, 10854, 10854, 21193, 21193, 12846, 60023, 11905, 18133, 18133, 120007, 18900, 60005, 19287, 10065, 10065, 10065, 10006, 10870, 18091, 18091, 10862, 10862, 10026, 12438, 10179, 10179, 18920, 10839, 10839, 19189, 10880, 10880, 128249, 128249, 12627, 19594, 10007, 10007, 100015, 100015, 10156, 10094, 19525, 18310, 12747, 12628, 10064, 19606, 48882, 19610, 19610, 90027, 11903, 12908, 10890, 10890, 10032, 10032, 12664, 12727, 12335, 12990, 60952, 10877, 68549, 68549, 10103, 10103, 19288, 13044, 19608, 60034, 10168, 10168, 52608, 12297, 12297, 10866, 10866, 70007, 12905, 18306, 10037, 12447, 10842, 10842]
            train_series = [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3, 4, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2]
            val_ids = [19119, 19237, 19236, 19236, 10157, 10157, 10879, 10879, 10879, 10184, 10045, 10045, 12901, 19521, 12531, 12728, 10855, 18896, 19454, 120985, 91343, 91343]
            val_series = [1, 1, 1, 2, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            test_ids = [10024, 10024, 10073, 10073, 10087, 10089, 10151, 10173, 10174, 10178, 10178, 10840, 10840, 10848, 10882, 10896, 12298, 12402, 12405, 12450, 12535, 12558, 12582, 12981, 12983, 12984, 13041, 18188, 18188, 18309, 18899, 19231, 19523, 19523, 19524, 19604, 19605, 19605, 49270, 52399, 52400, 52655, 90006, 100018, 120015, 128247]
            test_series = [1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        elif trial == 2: # 180 21 53
            train_ids = [12405, 12665, 12535, 12483, 12450, 10041, 12983, 10089, 12560, 19236, 19236, 19288, 10170, 12728, 10843, 10843, 10094, 21193, 21193, 10898, 10898, 12438, 19189, 18315, 10873, 18189, 10171, 10171, 12558, 19285, 19604, 10871, 10871, 12743, 10044, 10044, 10134, 19454, 52399, 10842, 10842, 10187, 10187, 10187, 10880, 10880, 12298, 19595, 18133, 18133, 10866, 10866, 10173, 12908, 19523, 19523, 49270, 12909, 10870, 10087, 18188, 18188, 10839, 10839, 12905, 12582, 12627, 12901, 19608, 68549, 68549, 60034, 19188, 19521, 10887, 19356, 19356, 19356, 128247, 58284, 10135, 48882, 12747, 19605, 19605, 90006, 10184, 128249, 128249, 90027, 10896, 10179, 10179, 10005, 19524, 100015, 100015, 18306, 91343, 91343, 19119, 10042, 10895, 12532, 18899, 12297, 12297, 60005, 10892, 12331, 91014, 300007, 12846, 11903, 13041, 10174, 52852, 89221, 12664, 10065, 10065, 10065, 18309, 10869, 10869, 10869, 10854, 10854, 18898, 18898, 10882, 10862, 10862, 19401, 10036, 10036, 12402, 52326, 10073, 10073, 10026, 17823, 17823, 19525, 10151, 10848, 19231, 10082, 12990, 52655, 12446, 12557, 10178, 10178, 10076, 10076, 108192, 10851, 12984, 19359, 12447, 10037, 60952, 10154, 10840, 10840, 10024, 10024, 12296, 10064, 10045, 10045, 12335, 120985, 18310, 100018, 10156, 10886, 10886, 13044]
            train_series = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 3, 1, 2, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 3, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1]
            val_ids = [19610, 19610, 120015, 60033, 10007, 10007, 18900, 10157, 10157, 18190, 52400, 10897, 10897, 12531, 10057, 10057, 11902, 19287, 12981, 10855, 60023]
            val_series = [1, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
            test_ids = [10006, 10032, 10032, 10066, 10066, 10103, 10103, 10124, 10131, 10136, 10168, 10168, 10176, 10176, 10186, 10856, 10867, 10877, 10879, 10879, 10879, 10885, 10890, 10890, 11905, 12583, 12628, 12630, 12727, 18091, 18091, 18132, 18314, 18511, 18511, 18896, 18920, 19233, 19233, 19237, 19286, 19286, 19286, 19286, 19594, 19606, 40012, 52608, 62046, 70007, 88648, 120007, 209135]
            test_series = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif trial == 3: # 180 19 55
            train_ids = [91343, 91343, 10856, 10854, 10854, 10176, 10176, 10897, 10897, 18091, 18091, 12532, 18188, 18188, 12630, 12901, 52608, 10898, 10898, 10005, 10174, 120007, 10037, 10006, 12984, 12727, 10087, 10073, 10073, 19286, 19286, 19286, 19286, 12447, 19119, 52655, 18899, 10869, 10869, 10869, 19233, 19233, 10839, 10839, 10041, 10178, 10178, 10134, 88648, 58284, 13044, 10887, 120015, 12402, 12582, 209135, 12747, 12628, 10171, 10171, 12665, 108192, 12296, 10892, 10151, 10186, 19525, 10870, 40012, 18309, 52399, 10076, 10076, 12483, 12983, 12535, 90027, 12298, 48882, 10173, 11903, 10851, 19188, 19231, 52400, 10136, 17823, 17823, 19454, 10890, 10890, 128249, 128249, 19606, 10103, 10103, 10168, 10168, 10885, 10131, 62046, 60952, 10855, 12743, 10842, 10842, 10867, 10179, 10179, 70007, 13041, 10089, 12560, 10036, 10036, 10866, 10866, 10848, 60023, 10032, 10032, 19287, 12558, 10064, 12335, 19610, 19610, 90006, 10879, 10879, 10879, 12446, 19604, 128247, 19237, 12450, 10026, 12583, 19594, 12728, 10880, 10880, 18132, 120985, 19359, 18920, 10840, 10840, 10024, 10024, 10066, 10066, 10896, 10082, 10877, 18511, 18511, 10094, 19401, 10843, 10843, 49270, 300007, 10135, 12990, 12627, 19288, 18310, 10187, 10187, 10187, 12981, 68549, 68549, 18896, 12846, 19523, 19523, 10154, 10124]
            train_series = [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 3, 4, 1, 1, 1, 1, 1, 2, 3, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1]
            val_ids = [19524, 12405, 89221, 11905, 12909, 19605, 19605, 19285, 12908, 12331, 10882, 19521, 18189, 10045, 10045, 18314, 100018, 10862, 10862]
            val_series = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2]
            test_ids = [10007, 10007, 10042, 10044, 10044, 10057, 10057, 10065, 10065, 10065, 10156, 10157, 10157, 10170, 10184, 10871, 10871, 10873, 10886, 10886, 10895, 11902, 12297, 12297, 12438, 12531, 12557, 12664, 12905, 18133, 18133, 18190, 18306, 18315, 18898, 18898, 18900, 19189, 19236, 19236, 19356, 19356, 19356, 19595, 19608, 21193, 21193, 52326, 52852, 60005, 60033, 60034, 91014, 100015, 100015]
            test_series = [1, 3, 1, 1, 2, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2]
        elif trial == 4: # 184 20 50 
            train_ids = [10877, 10026, 18920, 12630, 12557, 10037, 18188, 18188, 10186, 10042, 49270, 10851, 10131, 12664, 19231, 18511, 18511, 12905, 10882, 18306, 10179, 10179, 10885, 100015, 100015, 10890, 10890, 12558, 108192, 12560, 10855, 52326, 18315, 18190, 10187, 10187, 10187, 12483, 12582, 18896, 10024, 10024, 10006, 10843, 10843, 19236, 19236, 10032, 10032, 52852, 52400, 19608, 10174, 18314, 88648, 120015, 10869, 10869, 10869, 12628, 19286, 19286, 19286, 19286, 11905, 10124, 18309, 10895, 10076, 10076, 18899, 10044, 10044, 120985, 11902, 18091, 18091, 10073, 10073, 52655, 12984, 12665, 60033, 10896, 10840, 10840, 10184, 12535, 10897, 10897, 12438, 10045, 10045, 19401, 12532, 12901, 60005, 10898, 10898, 90027, 10007, 10007, 10880, 10880, 91014, 12727, 19359, 10871, 10871, 12981, 52608, 10057, 10057, 18900, 10157, 10157, 10156, 10892, 90006, 19188, 10065, 10065, 10065, 60034, 10064, 12297, 12297, 19454, 52399, 10173, 10168, 10168, 19119, 18898, 18898, 19285, 12402, 10082, 10170, 10867, 19605, 19605, 10856, 12531, 128247, 10848, 60023, 10136, 19594, 12405, 12728, 13044, 10103, 10103, 10879, 10879, 10879, 19233, 19233, 40012, 10094, 18133, 18133, 19356, 19356, 19356, 19606, 12846, 10151, 10087, 70007, 19595, 10066, 10066, 10886, 10886, 12627, 209135, 62046, 19237, 19604, 18310, 19189, 12298]
            train_series = [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 3, 1, 2, 1, 1, 1, 3, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1]
            val_ids = [21193, 21193, 58284, 100018, 12450, 19523, 19523, 19524, 12583, 12983, 10176, 10176, 10041, 10873, 18132, 10178, 10178, 10089, 13041, 120007]
            val_series = [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1]
            test_ids = [10005, 10036, 10036, 10134, 10135, 10154, 10171, 10171, 10839, 10839, 10842, 10842, 10854, 10854, 10862, 10862, 10866, 10866, 10870, 10887, 11903, 12296, 12331, 12335, 12446, 12447, 12743, 12747, 12908, 12909, 12990, 17823, 17823, 18189, 19287, 19288, 19521, 19525, 19610, 19610, 48882, 60952, 68549, 68549, 89221, 91343, 91343, 128249, 128249, 300007]
            test_series = [1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1]
        elif trial == 5:
            train_ids = [10877, 10026]
            train_series = [1, 1]
            val_ids = [21193, 58284]
            val_series = [1, 1]
            test_ids = [10005, 10036]
            test_series = [1, 1]  
    elif modality == 'CT':
        if trial == 0: # 53 6 14
            train_ids = [18918, 60042, 99401, 19407, 60932, 300006, 219506, 60022, 99198, 10851, 21425, 31283, 31283, 31283, 151926, 10898, 98894, 60032, 60032, 10023, 60959, 198864, 12484, 21618, 38266, 12533, 18310, 248850, 19284, 18133, 19115, 19591, 19591, 12516, 90827, 60008, 60008, 218631, 19289, 89222, 61126, 19234, 12585, 10848, 12629, 202433, 60013, 10160, 19282, 10094, 18707, 38109, 62047]
            train_series = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1]
            val_ids = [12485, 218778, 60015, 48439, 19521, 10856]
            val_series = [1, 1, 1, 1, 2, 1]
            test_ids = [10007, 10164, 10171, 10840, 10860, 19116, 42293, 60936, 88646, 90020, 100003, 131874, 179156, 248998]
            test_series = [2, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif trial == 1: # 51 6 16
            train_ids = [10164, 99198, 19284, 60932, 100003, 10860, 60015, 19521, 38266, 131874, 48439, 38109, 218631, 12516, 179156, 31283, 31283, 31283, 10094, 42293, 18707, 88646, 98894, 12585, 12484, 90827, 10840, 10898, 60959, 10023, 18918, 60008, 60008, 248998, 19115, 60936, 18310, 10007, 198864, 89222, 90020, 12485, 19234, 19407, 12629, 10848, 19116, 61126, 219506, 218778, 10160]
            train_series = [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
            val_ids = [202433, 10171, 19282, 60022, 248850, 60013]
            val_series = [1, 3, 1, 1, 1, 1]
            test_ids = [10851, 10856, 12533, 18133, 19289, 19591, 19591, 21425, 21618, 60032, 60032, 60042, 62047, 99401, 151926, 300006]
            test_series = [2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1]
        elif trial == 2: # 53 6 14
            train_ids = [60032, 60032, 38266, 61126, 90020, 12516, 19234, 131874, 19521, 300006, 248998, 60932, 10171, 42293, 248850, 60022, 19591, 19591, 60008, 60008, 21618, 10851, 12585, 219506, 18133, 10094, 12485, 151926, 10840, 19289, 198864, 31283, 31283, 31283, 19115, 18707, 99401, 38109, 179156, 10860, 98894, 90827, 100003, 62047, 12533, 60936, 10007, 10164, 21425, 202433, 18918, 10160, 19116]
            train_series = [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
            val_ids = [88646, 60015, 60042, 10856, 18310, 218778]
            val_series = [1, 1, 1, 1, 1, 1]
            test_ids = [10023, 10848, 10898, 12484, 12629, 19282, 19284, 19407, 48439, 60013, 60959, 89222, 99198, 218631]
            test_series = [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif trial == 3: # 53 7 13
            train_ids = [31283, 31283, 31283, 19115, 19284, 131874, 10860, 90020, 19282, 10094, 90827, 60936, 218631, 89222, 12585, 179156, 18918, 10007, 21618, 10171, 10840, 248998, 12533, 88646, 100003, 12629, 62047, 19407, 60042, 60015, 10851, 19289, 19116, 218778, 61126, 18133, 21425, 12484, 10164, 19521, 10023, 42293, 10898, 60008, 60008, 300006, 38266, 60013, 99401, 19591, 19591, 10848, 99198]
            train_series = [1, 2, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1]
            val_ids = [151926, 48439, 19234, 60959, 10856, 60032, 60032]
            val_series = [1, 1, 1, 1, 1, 1, 2]
            test_ids = [10160, 12485, 12516, 18310, 18707, 38109, 60022, 60932, 98894, 198864, 202433, 219506, 248850]
            test_series = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif trial == 4: # 51 6 16           
            train_ids = [198864, 100003, 12629, 19407, 60936, 60042, 62047, 19284, 12485, 10860, 10164, 10160, 10171, 10898, 38109, 19116, 21618, 60022, 19289, 218631, 42293, 131874, 18310, 60032, 60032, 60932, 12533, 12484, 10023, 19591, 19591, 60013, 89222, 151926, 300006, 179156, 19282, 10840, 18707, 90020, 99198, 202433, 248998, 99401, 10856, 18133, 10851, 98894, 10848, 21425, 219506]
            train_series = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1]
            val_ids = [10007, 48439, 12516, 248850, 88646, 60959]
            val_series = [2, 1, 1, 1, 1, 1]
            test_ids = [10094, 12585, 18918, 19115, 19234, 19521, 31283, 31283, 31283, 38266, 60008, 60008, 60015, 61126, 90827, 218778]
            test_series = [2, 1, 1, 1, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 1]
        elif trial == 5: 
            train_ids = [198864, 100003]
            train_series = [1, 1]
            val_ids = [10007, 48439]
            val_series = [2, 1]
            test_ids = [10094, 12585]
            test_series = [2, 1]        
    return list(train_ids), list(val_ids), list(test_ids), list(train_series), list(val_series), list(test_series)