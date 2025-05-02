#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from exams.exam_kits import exam_kits
import distutils.dir_util
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.transform import resize, rotate
from skimage import io
import tqdm
from utils import utils

def create_kits_dataset(output, ids, series, scheme, modality, size, anatomy):
    list_id = []
    folder = output+scheme+'/'
    distutils.dir_util.mkpath(folder)
    dim = 2 if modality == 'CT' else 1
    for idx, id_ in enumerate(tqdm.tqdm(ids)):
        exam = exam_kits(id_,series[idx],modality)
        print(exam.id)
        exam.normalize(modality)  
        ref = exam.CT if modality == 'CT' else exam.T2
        for xyz in range(ref.shape[dim]): 
            img, mask = extract_kits_slice(exam, xyz, modality, size, anatomy)
            list_id.append(modality+'-%0*d'%(3,id_)+'-%0*d-'%(2,series[idx])+anatomy+'-%0*d'%(3,xyz+1))
            io.imsave(folder+list_id[-1]+'-src.png', img)
            io.imsave(folder+list_id[-1]+'-mask.png', mask, check_contrast=False)
            if len(np.unique(mask))>1:
                io.imsave(folder+list_id[-1]+'-bound.png', img_as_ubyte(utils.boundaries(img, None, mask)))
        del exam
    np.save(output+'imgs-id-'+scheme+'.npy', list_id)
    del list_id
    
def create_kits_dataset_MT(output, ids, scheme, size):
    list_id = []
    folder = output+scheme+'/'
    distutils.dir_util.mkpath(folder)
    anatomy = 'LK+RK'
    dim = 2
    for idx, id_ in enumerate(tqdm.tqdm(ids)):
        exam = exam_kits(id_)
        print(exam.id)
        exam.normalize()
        for xyz in range(exam.CT.shape[dim]): 
            img, mask_LK, mask_RK = extract_kits_slice_MT(exam, xyz, size)
            list_id.append('CT-%0*d'%(5,id_)+anatomy+'-%0*d'%(3,xyz+1))
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
    
def extract_kits_slice(exam, xyz, size, anatomy):
    img = rotate(resize(np.squeeze(exam.CT.get_fdata()[:,:,xyz])[::-1,:], output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    if anatomy == 'BK':
        mask = rotate(resize(exam.BK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    elif anatomy == 'LK':
        mask = rotate(resize(exam.LK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    elif anatomy == 'RK':
        mask = rotate(resize(exam.RK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    min_greyscale, max_greyscale = np.percentile(img,(1,99)) 
    img = rescale_intensity(img, in_range=(min_greyscale,max_greyscale), out_range=(0,1))
    mask[np.where(mask>0)] = 255
    return img_as_ubyte(img), mask.astype(np.uint8)

def extract_kits_slice_MT(exam, xyz, size):
    img = rotate(resize(np.squeeze(exam.CT.get_fdata()[:,:,xyz])[::-1,:], output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    mask_LK = rotate(resize(exam.LK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    mask_RK = rotate(resize(exam.RK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    min_greyscale, max_greyscale = np.percentile(img,(1,99)) 
    img = rescale_intensity(img, in_range=(min_greyscale,max_greyscale), out_range=(0,1))
    mask_LK[np.where(mask_LK>0)] = 255
    mask_RK[np.where(mask_RK>0)] = 255
    return img_as_ubyte(img), mask_LK.astype(np.uint8), mask_RK.astype(np.uint8)

def kits_split():
    train_ids = [418, 120, 244, 86, 250, 147, 233, 95, 103, 87, 468, 501, 164, 423, 578, 217, 569, 577, 10, 464, 536, 288, 130, 198, 66, 11, 182, 226, 430, 459, 2, 277, 515, 168, 52, 72, 70, 199, 261, 31, 415, 275, 92, 547, 432, 80, 56, 230, 546, 485, 558, 20, 585, 192, 211, 6, 245, 239, 538, 477, 215, 90, 224, 22, 227, 176, 400, 426, 19, 142, 34, 586, 576, 65, 256, 427, 210, 97, 29, 216, 453, 225, 195, 208, 75, 446, 235, 157, 409, 406, 222, 580, 559, 63, 57, 203, 274, 96, 292, 408, 78, 259, 68, 509, 458, 109, 27, 197, 202, 258, 62, 8, 272, 500, 554, 30, 438, 46, 492, 403, 9, 154, 108, 110, 115, 104, 220, 91, 152, 519, 286, 123, 234, 184, 404, 282, 451, 238, 508, 53, 583, 60, 463, 163, 460, 126, 564, 206, 276, 439, 511, 497, 246, 498, 26, 433, 189, 134, 530, 413, 79, 255, 43, 54, 213, 482, 124, 299, 456, 58, 445, 489, 174, 25, 581, 196, 281, 248, 465, 218, 434, 297, 470, 229, 17, 469, 528, 178, 117, 490, 502, 507, 449, 503, 33, 266, 466, 44, 566, 127, 253, 537, 285, 562, 173, 457, 517, 283, 119, 187, 51, 207, 28, 488, 170, 549, 556, 435, 223, 480, 149, 156, 535, 185, 161, 491, 443, 136, 7, 14, 542, 186, 284, 55, 533, 512, 242, 141, 191, 3, 15, 440, 478, 575, 484, 232, 179, 461, 571, 231, 150, 510, 155, 165, 405, 525, 431, 294, 153, 532, 59, 240, 121, 73, 61, 523, 42, 534, 88, 454, 552, 475, 529, 118, 181, 414, 105, 527, 424, 560, 579, 260, 167, 137, 296, 518, 298, 158, 483, 425, 588, 169, 572, 544, 499, 450, 573, 67, 135, 21, 291, 89, 77, 131, 1, 582, 271, 236, 422, 144, 139, 40, 474, 526, 555, 268, 436, 287, 262, 200, 128, 24, 122, 473, 516, 38, 171, 565, 243, 545, 570, 401, 419, 201, 37, 402, 41, 125, 557, 180, 99, 273, 228, 251, 429, 289, 71, 505, 23, 543, 12, 83, 520, 252, 481, 531, 420, 140, 69, 522, 166, 190, 85, 524, 151, 48, 447, 129, 263, 18, 219, 574, 455, 132, 553, 16, 264, 214, 442, 437, 476, 247, 205, 587, 138, 539, 175, 45, 5, 270, 160, 421, 76, 548, 113, 237, 267, 36, 100, 412, 541, 417, 293, 471, 290, 212, 0, 513, 114, 407, 209, 172, 249, 452, 133, 106, 146, 416, 265, 506, 496, 102, 64, 280, 254, 411, 568, 159, 278, 84, 495, 550, 295, 112, 521, 269, 567, 493, 428, 145, 93, 561, 540, 177, 467, 148, 479, 143, 81, 47, 448, 487, 194, 410, 94, 50, 32, 101, 183, 162, 107, 241, 486, 204, 116, 188, 584, 551, 514, 279, 111, 472, 462, 441, 4, 504, 74]
    val_ids = [35, 221, 193, 13, 39, 49, 82, 257, 98, 494]
    # 444, 563 --> only 1 kidney
    return list(train_ids), list(val_ids)