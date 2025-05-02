import os
import nibabel as nib
import numpy as np

def preprocess_to_slices(img_dir, mask_dir, out_dir, modality="T2", mask_suffixes=("RK", "LK")):
    labeled = mask_dir is not None
    target_dir = os.path.join(out_dir, "labeled" if labeled else "unlabeled")
    os.makedirs(target_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')]

    for f in img_files:
        img_path = os.path.join(img_dir, f)
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata()

        # Déterminer l’axe des slices automatiquement
        shape = img_data.shape
        slice_axis = np.argmin(shape)  # on slice le long de l’axe le plus court

        # Rotation standard pour mise à plat : [H, W]
        if slice_axis == 0:
            slices = [np.rot90(img_data[i, :, :]) for i in range(shape[0])]
        elif slice_axis == 1:
            slices = [np.rot90(img_data[:, i, :]) for i in range(shape[1])]
        else:
            slices = [np.rot90(img_data[:, :, i]) for i in range(shape[2])]

        basename = f.replace(f"-{modality}.nii.gz", "")
        for i, s in enumerate(slices):
            np.save(os.path.join(target_dir, f"{basename}-slice-{i:03d}.npy"), s)

        if labeled:
            for suffix in mask_suffixes:
                mask_path = os.path.join(mask_dir, f"{basename}-{suffix}.nii.gz")
                if os.path.exists(mask_path):
                    mask_data = nib.load(mask_path).get_fdata()
                    mask_shape = mask_data.shape
                    if np.argmin(mask_shape) != slice_axis:
                        print(f"[⚠] Attention: mask {mask_path} shape {mask_shape} doesn't match image.")
                        continue

                    if slice_axis == 0:
                        slices = [np.rot90(mask_data[i, :, :]) for i in range(mask_shape[0])]
                    elif slice_axis == 1:
                        slices = [np.rot90(mask_data[:, i, :]) for i in range(mask_shape[1])]
                    else:
                        slices = [np.rot90(mask_data[:, :, i]) for i in range(mask_shape[2])]

                    for i, s in enumerate(slices):
                        np.save(os.path.join(target_dir, f"{basename}-slice-{i:03d}-mask-{suffix}.npy"), s)

    print(f"Préprocessing terminé dans : {target_dir}")



if __name__ == "__main__":
    preprocess_to_slices(
        img_dir="/home/hemery/code_halt_semi_supervised/data_halt_genkyst/labeled/T2",
        mask_dir="/home/hemery/code_halt_semi_supervised/data_halt_genkyst/labeled/mask",
        out_dir="/home/hemery/code_halt_semi_supervised/data_halt_genkyst/preprocessed",
        modality="T2"
    )

    # Et pour les non-labélisées :
    preprocess_to_slices(
        img_dir="/home/hemery/code_halt_semi_supervised/data_halt_genkyst/unlabeled/",
        mask_dir=None,
        out_dir="/home/hemery/code_halt_semi_supervised/data_halt_genkyst/preprocessed",
        modality="T2"
    )

