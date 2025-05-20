
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from skimage.transform import resize, rotate
from skimage.util import img_as_ubyte
import torch.nn.functional as F
from utils.utils import normalization_imgs, normalization_masks, min_max_normalization

class tiny_dataset(Dataset):
    def __init__(self, ids, series, labeled_flags, size=256, vgg=False, base_dir=None):
        self.entries = []
        self.size = size
        self.vgg = vgg
        self.base_dir = base_dir

        for id_, serie, labeled in zip(ids, series, labeled_flags):
            serie_str = f"{int(serie):02d}"

            if labeled:
                patient_id = f"{id_}-{serie_str}"
                t2_path = os.path.join(base_dir, "labeled", "T2", f"{patient_id}-T2.nii.gz")
                rk_path = os.path.join(base_dir, "labeled", "mask", f"{patient_id}-RK.nii.gz")
                lk_path = os.path.join(base_dir, "labeled", "mask", f"{patient_id}-LK.nii.gz")
            else:
                patient_id = f"{id_}-{serie_str}"
                t2_path = os.path.join(base_dir, "unlabeled", f"{patient_id}-T2.nii.gz")
                rk_path = lk_path = None


            if not os.path.exists(t2_path):
                print(f" Skipping: {t2_path}")
                continue

            try:
                img = nib.load(t2_path).get_fdata()
            except Exception as e:
                print(f" Failed to load image for {patient_id}: {e}")
                continue

            axis = np.argmin(img.shape)
            if axis == 0:
                slices = [np.rot90(img[i, :, :]) for i in range(img.shape[0])]
            elif axis == 1:
                slices = [np.rot90(img[:, i, :]) for i in range(img.shape[1])]
            else:
                slices = [np.rot90(img[:, :, i]) for i in range(img.shape[2])]

            if labeled and os.path.exists(rk_path) and os.path.exists(lk_path):
                rk = nib.load(rk_path).get_fdata()
                lk = nib.load(lk_path).get_fdata()
                if axis == 0:
                    mask_slices = [np.rot90((rk[i, :, :] + lk[i, :, :]) > 0) for i in range(rk.shape[0])]
                elif axis == 1:
                    mask_slices = [np.rot90((rk[:, i, :] + lk[:, i, :]) > 0) for i in range(rk.shape[1])]
                else:
                    mask_slices = [np.rot90((rk[:, :, i] + lk[:, :, i]) > 0) for i in range(rk.shape[2])]
            else:
                mask_slices = [None] * len(slices)

            for img_slice, mask_slice in zip(slices, mask_slices):
                self.entries.append({
                    "image": img_slice,
                    "mask": mask_slice,
                    "labeled": labeled 
                })

        print(f" Dataset total entries: {len(self.entries)}")
        print(f"  ↪ labeled: {sum(labeled_flags)} patients")
        print(f"  ↪ unlabeled: {len(labeled_flags) - sum(labeled_flags)} patients")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = entry["image"]
        mask = entry["mask"]

        img = img.astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        # normalisation 
        img = min_max_normalization(img.numpy())  
        img = torch.from_numpy(img).float()

        #print("TRAIN image stats - min:", img.min(), "max:", img.max(), "mean:", img.mean())
  
        
        img = F.interpolate(img.unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False).squeeze(0)

        # Si vous utilisez VGG, répétez l'image sur 3 canaux
        #if self.vgg:
            #img = img.repeat(3, 1, 1)

        if mask is not None:
            mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=(self.size, self.size), mode='nearest').squeeze(0)

            # Normalisation du masque
            #mask = normalization_masks(mask.numpy())  
            #mask = torch.from_numpy(mask).float()  # Convertir en tensor à nouveau après normalisation
  

            return img, mask
        else:
            return img
