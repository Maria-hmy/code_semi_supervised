import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from utils.utils import normalization_imgs, normalization_masks

class DatasetGenkystHalt(Dataset):
    def __init__(self, img_dir, mask_dir=None, size=256, is_labeled=True, modality="T2", anatomy="BK", vgg=False, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_labeled = is_labeled
        self.size = size
        self.modality = modality
        self.anatomy = anatomy
        self.vgg = vgg
        self.augment = augment
        self.resize = transforms.Resize((size, size))

        self.slice_index_map = []  # (img_path, mask_path, slice_idx)

        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(".nii.gz"):
                continue
            img_path = os.path.join(img_dir, fname)

            base = fname.replace("-T2.nii.gz", "")  # ex: 010134-01
            mask_RK_path = os.path.join(mask_dir, base + "-RK.nii.gz") if mask_dir else None
            mask_LK_path = os.path.join(mask_dir, base + "-LK.nii.gz") if mask_dir else None

            img = nib.load(img_path).get_fdata()
            depth = img.shape[1]  # slicing along axis 1

            for i in range(depth):
                self.slice_index_map.append((img_path, mask_RK_path, mask_LK_path, i))

    def __len__(self):
        return len(self.slice_index_map)

    def __getitem__(self, idx):
        img_path, mask_RK_path, mask_LK_path, slice_idx = self.slice_index_map[idx]

        img = nib.load(img_path).get_fdata()
        img_slice = np.rot90(np.squeeze(img[:, slice_idx, :]))

        if self.vgg:
            img_np = np.stack([img_slice]*3, axis=-1).astype(np.float32)
        else:
            img_np = img_slice[..., np.newaxis].astype(np.float32)

        img_pil = to_pil_image(img_np.squeeze())
        img_resized = self.resize(img_pil)
        img_resized = np.array(img_resized)
        if not self.vgg:
            img_resized = img_resized[..., np.newaxis]

        img_resized = normalization_imgs(img_resized)

        if self.is_labeled and os.path.exists(mask_RK_path) and os.path.exists(mask_LK_path):
            mask_RK = nib.load(mask_RK_path).get_fdata()
            mask_LK = nib.load(mask_LK_path).get_fdata()

            mask_slice = np.rot90(np.squeeze(mask_RK[:, slice_idx, :])) + \
                         np.rot90(np.squeeze(mask_LK[:, slice_idx, :]))

            mask_np = mask_slice[..., np.newaxis].astype(np.uint8)
            mask_pil = to_pil_image(mask_np.squeeze())
            mask_resized = self.resize(mask_pil)
            mask_resized = np.array(mask_resized)[..., np.newaxis]
            mask_resized = normalization_masks(mask_resized)
            return img_resized.swapaxes(2, 0), mask_resized.swapaxes(2, 0)
        else:
            return img_resized.swapaxes(2, 0)
        



import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class DatasetGenkystSlices(Dataset):
    def __init__(self, image_paths, mask_paths=None, vgg=False, target_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths if mask_paths is not None else [None] * len(image_paths)
        self.vgg = vgg
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = np.load(img_path).astype(np.float32)  # (H, W)
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        img = F.interpolate(img.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)

        if self.vgg:
            img = img.repeat(3, 1, 1)  # [3, 256, 256]

        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask = np.load(mask_path).astype(np.uint8)
            mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
            mask = F.interpolate(mask.unsqueeze(0), size=self.target_size, mode="nearest").squeeze(0)
            mask = (mask > 0).float()  # Binariser
            return img, mask
        else:
            return img




import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from skimage.transform import resize, rotate
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte

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
        img = F.interpolate(img.unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False).squeeze(0)
        if self.vgg:
            img = img.repeat(3, 1, 1)

        if mask is not None:
            mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=(self.size, self.size), mode='nearest').squeeze(0)
            return img, mask
        else:
            return img


    def normalize_img(self, img):
        return img / 255.0

    def normalize_mask(self, mask):
        return (mask > 127).astype(np.float32)

    def load_slice(self, entry):
        i = entry["slice_idx"]
        t2_img = nib.load(entry["t2_path"]).get_fdata()
        img = t2_img[:, i, :][::-1, :]
        img = rotate(resize(img, output_shape=(self.size, self.size), preserve_range=True), 90, preserve_range=True)
        img = rescale_intensity(img, in_range=(np.percentile(img, (1, 99))), out_range=(0, 1))
        img = img_as_ubyte(img)

        if entry["labeled"]:
            mask = np.zeros_like(img, dtype=np.uint8)
            if os.path.exists(entry["lk_path"]):
                lk = nib.load(entry["lk_path"]).get_fdata()[:, i, :][::-1, :] * 255
                lk = rotate(resize(lk, output_shape=(self.size, self.size), preserve_range=True), 90, preserve_range=True)
            else:
                lk = np.zeros_like(img)

            if os.path.exists(entry["rk_path"]):
                rk = nib.load(entry["rk_path"]).get_fdata()[:, i, :][::-1, :] * 255
                rk = rotate(resize(rk, output_shape=(self.size, self.size), preserve_range=True), 90, preserve_range=True)
            else:
                rk = np.zeros_like(img)

            mask = np.maximum(lk, rk)
            mask[mask > 0] = 255
            return img, mask.astype(np.uint8)
        else:
            return img, None


