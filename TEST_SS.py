import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
import torch.nn.functional as F
from nets.whichnet import whichnet
from sklearn.metrics import f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default="/home/hemery/code_halt_semi_supervised/data_halt_genkyst/test/images")
    parser.add_argument('--mask_dir', type=str, default="/home/hemery/code_halt_semi_supervised/data_halt_genkyst/test/masks")
    parser.add_argument('--output_dir', type=str, default="./results_test")
    parser.add_argument('--model', type=int, default=1)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--teacher', action='store_true', help="Use teacher model for inference")
    return parser.parse_args()

def get_mask_volume(mask_l_path, mask_r_path):
    if os.path.exists(mask_l_path) and os.path.exists(mask_r_path):
        m_l = nib.load(mask_l_path).get_fdata()
        m_r = nib.load(mask_r_path).get_fdata()
        return np.clip(m_l + m_r, 0, 1)
    return None

def save_nifti_volume(volume_np, reference_nifti_path, output_path):
    reference_nifti = nib.load(reference_nifti_path)
    nifti_img = nib.Nifti1Image(volume_np.astype(np.uint8), affine=reference_nifti.affine)
    nib.save(nifti_img, output_path)

def dice_score(pred, gt):
    pred_flat = (pred > 0.5).astype(np.uint8).flatten()
    gt_flat = (gt > 0.5).astype(np.uint8).flatten()
    return f1_score(gt_flat, pred_flat, zero_division=1)

def preprocess_slice(img_slice, target_size):
    img_slice = np.clip(img_slice, 0, np.percentile(img_slice, 99))
    img_slice = img_slice.astype(np.float32)
    img_slice /= (img_slice.max() + 1e-8)

    tensor = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    tensor = F.interpolate(tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return tensor  # [1, 1, size, size]

def postprocess_prediction(pred_tensor, original_shape):
    pred = torch.sigmoid(pred_tensor).squeeze().cpu()  # [H, W]
    pred_bin = (pred > 0.5).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    pred_resized = F.interpolate(pred_bin, size=original_shape, mode='nearest').squeeze().numpy()
    return pred_resized

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Modèle ===
    model, _ = whichnet(net_id=args.model, n_classes=1, img_size=args.size)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    image_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".nii.gz")])
    all_dice = []

    for fname in tqdm(image_files, desc="Inference"):
        subject_id = fname.replace("-T2.nii.gz", "")
        img_path = os.path.join(args.data_dir, fname)
        mask_l_path = os.path.join(args.mask_dir, subject_id + "-LK.nii.gz")
        mask_r_path = os.path.join(args.mask_dir, subject_id + "-RK.nii.gz")

        img_nii = nib.load(img_path)
        img_vol = img_nii.get_fdata()
        H, D, W = img_vol.shape  # (x, y, z)
        pred_vol = np.zeros((H, D, W), dtype=np.uint8)

        for slice_idx in range(D):  # coupe coronale
            img_slice = img_vol[:, slice_idx, :]
            tensor = preprocess_slice(img_slice, args.size).to(device)

            with torch.no_grad():
                output = model(tensor)
                pred_resized = postprocess_prediction(output, (H, W))
                pred_vol[:, slice_idx, :] = (pred_resized > 0.5).astype(np.uint8)

        # Sauvegarde
        out_path = os.path.join(args.output_dir, f"{subject_id}_pred.nii.gz")
        save_nifti_volume(pred_vol, img_path, out_path)

        # Évaluation
        gt_vol = get_mask_volume(mask_l_path, mask_r_path)
        if gt_vol is not None and gt_vol.shape == pred_vol.shape:
            dice = dice_score(pred_vol, gt_vol)
            all_dice.append(dice)
            print(f"Dice {subject_id}: {dice:.4f}")
        else:
            print(f"GT non trouvé ou mismatch pour {subject_id}")

    if all_dice:
        print(f"\n--- Résultat global ---")
        print(f"Dice moyen : {np.mean(all_dice):.4f}")

if __name__ == "__main__":
    main()
