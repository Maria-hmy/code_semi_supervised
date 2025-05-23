#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import logging
from utils.utils import normalization_imgs
import os

from utils.utils import force_to_257_N_257  

class exam_halt_prod:  

    def __init__(self, id_, serie, path, output, modality):
        self.id = str(id_)  # Supporte les IDs alphanumériques
        self.serie = serie
        self.path = path
        self.output = output
        self.modality = modality
        self.exam_upload()  # Charge l'examen
        self.print_info()   # Affiche les informations de l'examen

    def exam_upload(self):
        filename = f"{self.id}-{self.serie:02d}-{self.modality}.nii.gz"
        filepath = os.path.join(self.path, filename)
        os.makedirs(self.output, exist_ok=True)

        if self.modality == 'T2':
            self.T2 = nib.as_closest_canonical(nib.load(filepath))
        elif self.modality == 'CT':
            self.CT = nib.as_closest_canonical(nib.load(filepath))

    def print_info(self):
        logging.basicConfig(level=logging.INFO, format='\n %(levelname)s: %(message)s')
        logging.info(f'''exam {self.id} uploaded:
        serie:         {self.serie}
        modality:      {self.modality}''')


