import sys
import matplotlib.pyplot as plt
import os
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from medpy.metric.binary import dc, assd, hd
import pandas as pd
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import nibabel as nib
from skimage.transform import rotate, resize

from utils.utils import prob2mask, get_array_affine_header, getLargestConnectedArea
from datasets.test_dataset import tiny_dataset_test
from nets.whichnet import whichnet

def get_args():
    parser = argparse.ArgumentParser(description='Segmentation rénale semi_supervisée  T2 HALT')
    parser.add_argument('-p', '--patient', type=str, required=True, help='ID patient (ex: B9309076)')
    parser.add_argument('-s', '--serie', type=int, required=True, help='ID série (ex: 2)')
    parser.add_argument('-o', '--output', type=str, default='./outputs/', help='Dossier de sortie')
    parser.add_argument('--gt_dir', type=str, default='./GT', help="Dossier contenant les GT (masques)")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    # === Paramètres modèle ===
    net_id = 1
    n_classes = 1
    size = 256
    modality = 'T2'

    # === Chargement du modèle ===
    net, vgg = whichnet(net_id, n_classes, size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.load_state_dict(torch.load('student_epoch_200.pth', map_location=device))
    net.eval()
    logging.info(f"Modèle chargé sur {device}")

    # === Chargement des données ===
    dataset = tiny_dataset_test(
        id_=args.patient,
        serie=args.serie,
        size=size,
        path='',  # ignoré dans le dataset modifié
        output=args.output,
        modality=modality,
        vgg=vgg
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    array, affine, header = get_array_affine_header(dataset, modality)

    
    # === Prédiction ===

   

    png_dir = os.path.join(args.output, f"{args.patient}-{args.serie:02d}-png_probs")
    os.makedirs(png_dir, exist_ok=True)

    with torch.no_grad():
        t2_data = dataset.exam.T2_data  # image normalisée

        depth = t2_data.shape[1]

        for idx, data in enumerate(loader):
            image = data.to(device=device, dtype=torch.float32)

            print(f"[Slice {idx}] Unique image values: {np.unique(image.cpu().numpy())}")



            pred = net(image)  # [1, 1, H, W]


            print(f"[Slice {idx}] Unique pred values: {np.unique(pred.cpu().numpy())}")



            
            prob = torch.sigmoid(pred).squeeze().cpu().numpy()  # [H, W]

            print(f"[Slice {idx}] Unique sigmoid(pred) values: {np.unique(prob)}")


            # --- Resize vers la forme originale ---
            target_shape = (t2_data.shape[0], t2_data.shape[2])
            prob_resized = resize(
                prob,
                target_shape,
                order=1,  # interpolation bilinéaire
                preserve_range=True,
                anti_aliasing=True
            )

            
            array[:, idx, :] = (prob_resized[::-1, :] > 0.5).astype(np.uint8)


            # svg png
            png_path = os.path.join(png_dir, f"prob_{idx:03d}.png")
            plt.imsave(png_path, prob_resized, cmap='gray', vmin=0, vmax=1)





    # === Post-traitement ===
   
    nib.save(nib.Nifti1Image(array.astype(np.uint16), affine, header),
             f"{args.output}/{args.patient}-{args.serie:02d}-prediction.nii.gz")

    # === MÉTRIQUES D'ÉVALUATION ===
    
    gt_lk_path = os.path.join(args.gt_dir, f"{args.patient}-{args.serie:02d}-LK.nii.gz")
    gt_rk_path = os.path.join(args.gt_dir, f"{args.patient}-{args.serie:02d}-RK.nii.gz")

    if not os.path.exists(gt_lk_path) or not os.path.exists(gt_rk_path):
        logging.warning(f"GT manquant : {gt_lk_path} ou {gt_rk_path}")
    else:
        gt_lk = nib.load(gt_lk_path).get_fdata().astype(np.uint8)
        gt_rk = nib.load(gt_rk_path).get_fdata().astype(np.uint8)
        gt_array = gt_lk + gt_rk
        gt_array[gt_array > 0] = 1  # binarisation


        if gt_array.shape != array.shape:
            logging.warning("GT et prédiction ont des dimensions différentes")
        else:
            try:
                dice = dc(array, gt_array)
                hausdorff = hd(array, gt_array)
                assd_score = assd(array, gt_array)
            except Exception as e:
                logging.warning(f"Erreur calcul métriques : {e}")
                dice, hausdorff, assd_score = -1, -1, -1

            # === Écriture CSV des métriques ===
            metrics_path = os.path.join(args.output, "metrics_summary.csv")
            results = {
                "Patient": args.patient,
                "Serie": args.serie,
                "Dice": dice,
                "HD": hausdorff,
                "ASSD": assd_score,
            }

            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
            else:
                df = pd.DataFrame([results])

            df.to_csv(metrics_path, index=False)
            print(f" Métriques sauvegardées dans {metrics_path}")

    logging.info("Segmentation terminée et fichiers enregistrés.")
