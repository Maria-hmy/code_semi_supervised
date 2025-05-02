

from torch.utils.data.dataset import Dataset
from skimage import io
import numpy as np
import torch
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte
import os
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, affine
from exams.exam_halt import exam_halt
from exams.exam_halt_prod import exam_halt_prod
from manage.manage_genkyst import test_normalization
from utils.utils import normalization_imgs, normalization_masks



class tiny_dataset_test(Dataset):
    ''' Dataset pour prédiction sur une seule IRM T2 annotée ou non, sans CT '''

    def __init__(self, id_, serie, size, path, output, modality, vgg: bool = False):
        self.id = id_
        self.serie = serie
        self.size = size
        self.vgg = vgg
        self.output = output
        self.modality = 'T2'
        self.path = "/home/hemery/code_halt_semi_supervised/data_halt_genkyst/labeled/T2"  # fixe chemin local T2 uniquement

        if modality != 'T2':
            raise ValueError("Seul le mode 'T2' est supporté dans tiny_dataset_halt_prod.")

        self.exam = exam_halt_prod(self.id, self.serie, self.path, self.output, self.modality)
        self.exam.normalize()

    def __len__(self):
        return self.exam.T2.shape[1]  

    
    
    def __getitem__(self, idx: int):
        img = test_normalization(self.exam, idx, self.size, self.modality)

        if self.vgg:
            img_ = np.stack([img] * 3, axis=-1)
        else:
            img_ = np.expand_dims(img, axis=-1)

        img_tensor = torch.from_numpy(img_.transpose(2, 0, 1))
        return img_tensor





