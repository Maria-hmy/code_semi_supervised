#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Function
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure, _ni_support

def dc(result, reference):
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 1.0
    return dc

def sensitivity(result, reference):
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)
    try:
        sensitivity = tp / float(tp + fn)
    except ZeroDivisionError:
        sensitivity = 0.0
    return sensitivity

def specificity(result, reference):
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)
    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0
    return specificity

def jc(result, reference):
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)
    try:
        jc = float(intersection) / float(union)
    except ZeroDivisionError:
        jc = 1.0
    return jc

def ravd(result, reference):
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)
    if 0 == vol2:
        raise RuntimeError('The second supplied array does not contain any binary object.')
    return (vol1 - vol2) / float(vol2)

def surface_distances(result, reference, voxelspacing=None, connectivity=1):
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=float)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    footprint = generate_binary_structure(result.ndim, connectivity)
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    return sds

def assd(result, reference, voxelspacing=None, connectivity=1):
    s1 = surface_distances(result, reference, voxelspacing, connectivity)
    s2 = surface_distances(reference, result, voxelspacing, connectivity)
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    return (ms1+ms2)/2

def hd(result, reference, voxelspacing=None, connectivity=1):
    hd1 = surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = surface_distances(reference, result, voxelspacing, connectivity).max()
    return max(hd1, hd2)

class DiceCoeff(Function):
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

def dice_coeff(input, target):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
    return s / (i + 1)

def assessment(result, groundtruth, domain, factor, label=1):
    dice = 100.*dc(result.get_fdata().astype(bool), groundtruth.get_fdata().astype(bool))
    sens = 100.*sensitivity(result.get_fdata().astype(bool), groundtruth.get_fdata().astype(bool))
    spec = 100.*specificity(result.get_fdata().astype(bool), groundtruth.get_fdata().astype(bool))
    jacc = 100.*jc(result.get_fdata().astype(bool), groundtruth.get_fdata().astype(bool))
    xspacing = abs(result.get_qform()[0,0])
    yspacing = abs(result.get_qform()[1,1])    
    zspacing = abs(result.get_qform()[2,2])   
    avd = np.abs(ravd(result.get_fdata().astype(bool), groundtruth.get_fdata().astype(bool)))
    connectivity = 1
    if len(np.unique(result.get_fdata().astype(bool)[::factor,::factor,::factor])) > 1:
        assd_ = assd(result.get_fdata().astype(bool)[::factor,::factor,::factor], groundtruth.get_fdata().astype(bool)[::factor,::factor,::factor], (xspacing*factor,yspacing*factor,zspacing*factor), connectivity)
        mssd_ = hd(result.get_fdata().astype(bool)[::factor,::factor,::factor], groundtruth.get_fdata().astype(bool)[::factor,::factor,::factor], (xspacing*factor,yspacing*factor,zspacing*factor), connectivity)
    else:
        assd_ = -1
        mssd_ = -1
    return np.array([dice, sens, spec, jacc, avd, assd_, mssd_])