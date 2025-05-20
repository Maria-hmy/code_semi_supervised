#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.exposure import rescale_intensity
from skimage.measure import label
import nibabel
import matplotlib.pyplot as plt

def boundaries(img, pred, groundtruth):
    img = rescale_intensity(img, in_range=(np.min(img),np.max(img)), out_range=(0,1))
    if type(pred) == np.ndarray and type(groundtruth) == np.ndarray:
        out = mark_boundaries(img, groundtruth, color=(0, 1, 0), background_label=4)
        out = mark_boundaries(out, pred, color=(1, 0, 0), background_label=2)
    else:
        if type(pred) == np.ndarray:
            out = mark_boundaries(img, pred, color=(1, 0, 0), background_label=2)
        if type(groundtruth) == np.ndarray:
            out = mark_boundaries(img, groundtruth, color=(0, 1, 0), background_label=4)            
    return out

def print_ods_(scores, test_ids, output, name):
    resfile = open(output+name, "a")
    resfile.write('exam\t'+'dice\t'+'sens\t'+'spec\t'+'jacc\t'+'avd\t'+'assd\t'+'mssd\t\n')
    for index, id_ in enumerate(test_ids):
        resfile.write('%0*d'%(3,id_)+'\t')
        for idx in range(scores.shape[1]):
            resfile.write(str('%.3f'%scores[index,idx]).replace(".", ",")+'\t')
        resfile.write('\n')
    resfile.write('mean\t') 
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.mean(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\nstd\t')
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.std(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\n')
    resfile.close()
    
def print_ods(scores, test_ids, test_series, output, name):
    resfile = open(output+name, "a")
    resfile.write('exam\t'+'serie\t'+'dice\t'+'sens\t'+'spec\t'+'jacc\t'+'avd\t'+'assd\t'+'mssd\t\n')
    for index, id_ in enumerate(test_ids):
        resfile.write('%0*d'%(3,id_)+'\t')
        resfile.write('%0*d'%(2,test_series[index])+'\t')        
        for idx in range(scores.shape[1]):
            resfile.write(str('%.3f'%scores[index,idx]).replace(".", ",")+'\t')
        resfile.write('\n')
    resfile.write('mean\t\t') 
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.mean(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\nstd\t\t')
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.std(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\n')
    resfile.close()

# =======
def metahep_resec_print_ods(scores, test_ids, test_series, output, name):
    resfile = open(output+name, "a")
    resfile.write('exam\t'+'serie\t'+'dice\t'+'sens\t'+'spec\t'+'jacc\t'+'avd\t'+'assd\t'+'mssd\t\n')
    for index, id_ in enumerate(test_ids):
        resfile.write('%0*d'%(3,id_)+'\t')
        resfile.write('%0*d'%(2,test_series[index])+'\t')
        for idx in range(scores.shape[1]):
            resfile.write(str('%.3f'%scores[index,idx]).replace(".", ",")+'\t')
        resfile.write('\n')
    resfile.write('mean\t\t') 
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.mean(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\nstd\t\t')
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.std(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\n')
    resfile.close()
# =======
    
def normalization_imgs(imgs):
    ''' centering and reducing data structures '''
    imgs = imgs.astype(np.float32, copy=False)
    mean = np.mean(imgs) # mean for data centering
    std = np.std(imgs) # std for data normalization
    if np.int32(std) != 0:
        imgs -= mean
        imgs /= std
    return imgs

def normalization_masks(imgs_masks):
    imgs_masks = imgs_masks.astype(np.float32, copy=False)
    imgs_masks /= 255.
    imgs_masks = imgs_masks.astype(np.uint8)
    return imgs_masks


def min_max_normalization(img):
    """Applique une normalisation min-max sur l'image."""
    img_min = img.min()
    img_max = img.max()
    if img_max != img_min:
        img = (img - img_min) / (img_max - img_min)  # Normalisation min-max
    return img


def get_array_affine_header(test_dataset, modality):
    if modality == 'T2':
        array = np.zeros(test_dataset.exam.T2.shape, dtype=np.uint16)
        affine, header = test_dataset.exam.T2.affine, test_dataset.exam.T2.header
    elif modality == 'CT':
        array = np.zeros(test_dataset.exam.CT.shape, dtype=np.uint16)
        affine, header = test_dataset.exam.CT.affine, test_dataset.exam.CT.header        
    return array, affine, header


import numpy as np

import nibabel as nib


import numpy as np

def force_to_257_N_257(volume, affine, name=""):
    if volume.shape[0] == 512 and volume.shape[1] == 512:
        print(f"[{name}] Transformation appliquée : {volume.shape} -> ({volume.shape[0]}, {volume.shape[2]}, {volume.shape[1]})")

        volume = np.transpose(volume, (0, 2, 1))  # y <-> z
        volume = volume[:, ::-1, :]               # Inversion de l'axe Z (ancien Y)
        volume = volume[::-1, :, :]               # Inversion de l'axe X (gauche-droite)
        volume = volume[:, :, ::-1]               # 🆕 Inversion de l'axe S/I (nouvelle Z après transpo)

        affine_fixed = affine.copy()
        affine_fixed[:3, [1, 2]] = affine_fixed[:3, [2, 1]]
        affine_fixed[:3, 2] = -affine_fixed[:3, 2]
        affine_fixed[:3, 0] = -affine_fixed[:3, 0]
        # affine_fixed[:3, 1] = -affine_fixed[:3, 1]  # ❌ Supprimer cette ligne

        return volume, affine_fixed
    else:
        print(f"[{name}] Aucune transformation nécessaire : {volume.shape}")
        return volume, affine











def img_init(img):
    return nibabel.Nifti1Image(img.get_fdata().astype(np.float32), affine=img.affine, header=img.header)

def img_zero(mask):
    return nibabel.Nifti1Image(np.zeros(shape=mask.shape).astype(np.float32), affine=mask.affine, header=mask.header)

def img_init_array(img, array):
    return nibabel.Nifti1Image(array.astype(np.float32), affine=img.affine, header=img.header)
    
def mask_init(mask):
    return nibabel.Nifti1Image(mask.get_fdata().astype(np.uint8), affine=mask.affine, header=mask.header)

def mask_zero(mask):
    return nibabel.Nifti1Image(np.zeros(shape=mask.shape).astype(np.uint8), affine=mask.affine, header=mask.header)

def mask_init_array(img, array):
    return nibabel.Nifti1Image(array.astype(np.uint8), affine=img.affine, header=img.header)

def getLargestConnectedArea(segmentation):
    if len(np.unique(segmentation)) == 1:
        return segmentation
    else:
        labels = label(segmentation,connectivity=1)
        unique, counts = np.unique(labels, return_counts=True)
        list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
        largest=max(list_seg, key=lambda x:x[1])[0]
        labels_max=(labels == largest).astype(int)
        return labels_max
    
def get2LargestConnectedAreas(segmentation, separate=False):
    if len(np.unique(segmentation)) == 1:
        return segmentation
    else:
        labels = label(segmentation,connectivity=1)
        unique, counts = np.unique(labels, return_counts=True)
        list_seg=list(zip(unique, counts))[1:]
        largest=max(list_seg, key=lambda x:x[1])[0]
        labels_max1=(labels == largest)
        labels[np.where(labels == largest)] = 0
        unique, counts = np.unique(labels, return_counts=True)
        list_seg=list(zip(unique, counts))[1:]
        largest=max(list_seg, key=lambda x:x[1])[0]
        labels_max2=(labels == largest)
        if separate:
            return labels_max1, labels_max2
        else:
            return labels_max1+labels_max2
    
def prob2mask(prob):
    mask = prob.squeeze().cpu().numpy()
    mask[mask<0.5] = 0
    mask[mask>=0.5] = 1
    return mask.swapaxes(0,1).astype(np.uint8)   

def volume_concordance_figure(true, automated, output, sup=-1):
    if sup == -1:
        sup = max(max(true),max(automated))+100
    plt.plot([0,sup], [0,sup],'b')
    plt.plot(true, automated, 'ro')
    plt.grid()
    plt.xlabel('true TKV (mL)')
    plt.ylabel('automated TKV (mL)')
    plt.xlim([0,sup])
    plt.ylim([0,sup])
    plt.legend(['perfect concordance'], loc='upper left')
    plt.savefig(output+'vol-concordance.pdf')
    plt.close()
    
def agreement_figure(true, automated, output, sup=-1):
    if sup == -1:
        sup = max(max(true),max(automated))+100
    true = np.array(true)
    automated = np.array(automated)
    plt.plot([0,sup], [0,0],'b')
    plt.plot(true, (automated-true)*100/true, 'ro')
    plt.grid()
    plt.xlabel('true TKV (mL)')
    plt.ylabel('difference (%)')
    plt.xlim([0,sup])
    plt.ylim([-100,100])
    plt.legend(['perfect agreement'], loc='upper right')
    plt.savefig(output+'vol-agreement.pdf')
    plt.close()
    
def all_volume_concordance_figure(true, automated_BO, automated_IND, automated_MT, output, supx=-1, supy=-1):
    if supx == -1 and supy == -1:
        sup = max(max(true),max(automated_BO),max(automated_IND),max(automated_MT))+100
        supx, supy = sup, sup
    plt.plot([0,max(supx,supy)], [0,max(supx,supy)],'black')
    plt.scatter(true, automated_BO, c='r', marker='*', alpha=0.7)
    plt.scatter(true, automated_IND, c='b', marker='*', alpha=0.7)
    plt.scatter(true, automated_MT, c='g', marker='*', alpha=0.7)
    plt.grid()
    plt.xlabel('true TKV (mL)')
    plt.ylabel('automated TKV (mL)')
    plt.xlim([300,supx])
    plt.ylim([300,supy])
    plt.legend(['perfect concordance', 'BO', 'IND', 'DT'], loc='upper left')
    plt.savefig(output+'vol-concordance.pdf')
    plt.close()
    
def all_volume_concordance_figure_network_level(true, automated_v19pUNet, automated_TransUNet, automated_medT, automated_Segmenter, automated_SwinUNetV2, output, supx=-1, supy=-1):
    if supx == -1 and supy == -1:
        sup = max(max(true),max(automated_v19pUNet),max(automated_TransUNet),max(automated_medT),max(automated_Segmenter),max(automated_SwinUNetV2))+100
        supx, supy = sup, sup
    plt.plot([0,max(supx,supy)], [0,max(supx,supy)],'black')
    plt.scatter(true, automated_v19pUNet, c='r', marker='*', alpha=0.7)
    plt.scatter(true, automated_TransUNet, c='y', marker='*', alpha=0.7)
    plt.scatter(true, automated_medT, c='g', marker='*', alpha=0.7)
    plt.scatter(true, automated_Segmenter, c='c', marker='*', alpha=0.7)
    plt.scatter(true, automated_SwinUNetV2, c='m', marker='*', alpha=0.7)
    plt.grid()
    plt.xlabel('true TKV (mL)')
    plt.ylabel('automated TKV (mL)')
    plt.xlim([300,supx])
    plt.ylim([300,supy])
    plt.legend(['perfect concordance', 'v19pUNet', 'TransUNet', 'medT', 'Segmenter', 'SwinUNetV2'], loc='upper left')
    plt.savefig(output+'vol-concordance.pdf')
    plt.close()
    
def all_agreement_figure(true, automated_BO, automated_IND, automated_MT, output, sup=-1):
    if sup == -1:
        sup = max(true)+100
    true = np.array(true)
    automated_BO, automated_IND, automated_MT = np.array(automated_BO), np.array(automated_IND), np.array(automated_MT)
    plt.plot([0,sup], [0,0],'b')
    plt.scatter(true, (automated_BO-true)*100/true, c='r', marker='*', alpha=0.7)
    plt.scatter(true, (automated_IND-true)*100/true, c='b', marker='*', alpha=0.7)
    plt.scatter(true, (automated_MT-true)*100/true, c='g', marker='*', alpha=0.7)
    plt.grid()
    plt.xlabel('true TKV (mL)')
    plt.ylabel('difference (%)')
    plt.xlim([300,sup])
    plt.ylim([-50,70])
    plt.legend(['perfect agreement', 'BO', 'IND', 'DT'], loc='lower right')
    plt.savefig(output+'vol-agreement.pdf')
    plt.close()
    
def all_agreement_figure_network_level(true, automated_v19pUNet, automated_TransUNet, automated_medT, automated_Segmenter, automated_SwinUNetV2, output, sup=-1):
    if sup == -1:
        sup = max(true)+100
    true = np.array(true)
    automated_v19pUNet, automated_TransUNet, automated_medT, automated_Segmenter, automated_SwinUNetV2 = np.array(automated_v19pUNet), np.array(automated_TransUNet), np.array(automated_medT), np.array(automated_Segmenter), np.array(automated_SwinUNetV2)
    plt.plot([0,sup], [0,0],'b')
    plt.scatter(true, (automated_v19pUNet-true)*100/true, c='r', marker='*', alpha=0.7)
    plt.scatter(true, (automated_TransUNet-true)*100/true, c='y', marker='*', alpha=0.7)
    plt.scatter(true, (automated_medT-true)*100/true, c='g', marker='*', alpha=0.7)
    plt.scatter(true, (automated_Segmenter-true)*100/true, c='c', marker='*', alpha=0.7)
    plt.scatter(true, (automated_SwinUNetV2-true)*100/true, c='m', marker='*', alpha=0.7)
    plt.grid()
    plt.xlabel('true TKV (mL)')
    plt.ylabel('difference (%)')
    plt.xlim([300,sup])
    plt.ylim([-50,70])
    plt.legend(['perfect agreement', 'v19pUNet', 'TransUNet', 'medT', 'Segmenter', 'SwinUNetV2'], loc='lower right')
    plt.savefig(output+'vol-agreement.pdf')
    plt.close()
    
def compute_TKV(niiBKmask):
    xspacing, yspacing, zspacing = abs(niiBKmask.get_qform()[0,0]), abs(niiBKmask.get_qform()[1,1]), abs(niiBKmask.get_qform()[2,2])
    spacing = xspacing*yspacing*zspacing
    return float(spacing)*len(np.where(niiBKmask.get_fdata()>0.)[0])/1000.