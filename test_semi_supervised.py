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

from utils.utils import prob2mask, get_array_affine_header, get2LargestConnectedAreas
from datasets.test_dataset import tiny_dataset_test
from nets.whichnet import whichnet
from utils.utils import force_to_257_N_257  

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
    net.load_state_dict(torch.load('039_epoch_best_model_teacher.pth', map_location=device))
    net.eval()
    logging.info(f"Modèle chargé sur {device}")



#vérif
    state_dict = torch.load('039_epoch_best_model_teacher.pth')
    missing, unexpected = net.load_state_dict(state_dict, strict=False)

    print("Paramètres manquants :", missing)
    print("Paramètres inattendus :", unexpected)


    # === Chargement des données ===
    dataset = tiny_dataset_test(
        id_=args.patient,
        serie=args.serie,
        size=size,
        path='',  
        output=args.output,
        modality=modality,
        vgg=vgg
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    array, affine, header = get_array_affine_header(dataset, modality)

    


    # === Prédiction ===

    with torch.no_grad():
        t2_data = dataset.exam.T2.get_fdata()

        
        depth = t2_data.shape[1]

        for idx, data in enumerate(loader):
            image = data.to(device=device, dtype=torch.float32)
            pred = net(image)
            #print(f"Input tensor - min: {image.min()}, max: {image.max()}, mean: {image.mean()}")


            prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            #print(f"Prediction sigmoid - min: {prob.min()}, max: {prob.max()}, mean: {prob.mean()}")



            target_shape = (t2_data.shape[0], t2_data.shape[2])
            prob_resized = resize(
                prob,
                target_shape,
                order=1,
                preserve_range=True,
                anti_aliasing=True
            )

            array[:, idx, :] = np.rot90(prob_resized, -1)[::-1, :] > 0.5

        # Garder uniquement les 2 plus grandes régions connectées sur tout le volume 3D
        #array = get2LargestConnectedAreas(array.astype(np.uint8))
    

    # === Post-traitement ===
    nib.save(nib.Nifti1Image(array.astype(np.uint16), affine, header),
            os.path.join(args.output, f"{args.patient}-{args.serie:02d}-prediction.nii.gz"))

    # === Sauvegarde de l'image T2 normalisée ===
    t2_output_path = os.path.join(args.output, f"{args.patient}-{args.serie:02d}-T2.nii.gz")
    nib.save(nib.Nifti1Image(t2_data, affine, header), t2_output_path)

    # === MÉTRIQUES D'ÉVALUATION ===
    gt_lk_path = os.path.join(args.gt_dir, f"{args.patient}-{args.serie:02d}-LK.nii.gz")
    gt_rk_path = os.path.join(args.gt_dir, f"{args.patient}-{args.serie:02d}-RK.nii.gz")

    if not os.path.exists(gt_lk_path) or not os.path.exists(gt_rk_path):
        logging.warning(f"GT manquant : {gt_lk_path} ou {gt_rk_path}")
    else:
        gt_lk = nib.load(gt_lk_path).get_fdata()
        gt_rk = nib.load(gt_rk_path).get_fdata()
        gt_lk, affine_lk = force_to_257_N_257(gt_lk, affine, name="GT_LK")
        gt_rk, affine_rk = force_to_257_N_257(gt_rk, affine, name="GT_RK")

       


        gt_array = (gt_lk > 0.5).astype(np.uint8) + (gt_rk > 0.5).astype(np.uint8)
        gt_array[gt_array > 0] = 1

        print("GT LK dtype:", gt_lk.dtype)
        print("GT LK shape:", gt_lk.shape)
        print("GT LK max:", np.max(gt_lk))
        print(f"Forme de la prédiction : {array.shape}")
        print(f"Forme de la GT combinée : {gt_array.shape}")
        print(f"Nombre de voxels prédits positifs : {np.sum(array)}")
        print(f"Nombre de voxels GT positifs : {np.sum(gt_array)}")

        if gt_array.shape != array.shape:
            logging.warning(f"Shape mismatch: prédiction {array.shape} vs GT {gt_array.shape}")
            dice, hausdorff, assd_score = -1, -1, -1
        elif np.sum(array) == 0 or np.sum(gt_array) == 0:
            logging.warning("GT ou prédiction vide (aucun voxel positif)")
            dice, hausdorff, assd_score = -1, -1, -1
        else:
            try:
                pred_bin = array.astype(np.uint8)
                gt_bin = gt_array.astype(np.uint8)
                print("Dice input unique values:", np.unique(pred_bin), np.unique(gt_bin))
                print("Array dtype:", pred_bin.dtype, "GT dtype:", gt_bin.dtype)
                dice = dc(pred_bin, gt_bin)
                hausdorff = hd(pred_bin, gt_bin)
                assd_score = assd(pred_bin, gt_bin)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logging.warning(f"Erreur calcul métriques : {e}")
                dice, hausdorff, assd_score = -1, -1, -1

        nib.save(nib.Nifti1Image(gt_array, affine, header),
                os.path.join(args.output, f"{args.patient}-{args.serie:02d}-GT.nii.gz"))

        # === Écriture CSV des métriques ===
        
    metrics_path = os.path.join(args.output, "metrics_summary.csv")
    results = {
        "Patient": args.patient,
        "Serie": args.serie,
        "Dice": dice,
        "HD": hausdorff,
        "ASSD": assd_score,
    }

    # Lire ancien CSV s’il existe, sinon partir de zéro
    if os.path.exists(metrics_path):
        df_existing = pd.read_csv(metrics_path)
        df_existing = df_existing[df_existing["Patient"] != "MOYENNE"]  # Supprimer ancienne ligne moyenne
    else:
        df_existing = pd.DataFrame()

    # Ajouter les résultats du patient courant
    df_current = pd.DataFrame([results])
    df_all = pd.concat([df_existing, df_current], ignore_index=True)

    # Recalculer moyenne globale uniquement sur les lignes valides
    valid_df = df_all[(df_all["Dice"] != -1) & (df_all["HD"] != -1) & (df_all["ASSD"] != -1)]

    if not valid_df.empty:
        avg_row = {
            "Patient": "MOYENNE",
            "Serie": "",
            "Dice": valid_df["Dice"].mean(),
            "HD": valid_df["HD"].mean(),
            "ASSD": valid_df["ASSD"].mean(),
        }
        df_all = pd.concat([df_all, pd.DataFrame([avg_row])], ignore_index=True)

    # Sauvegarder le tout
    df_all.to_csv(metrics_path, index=False)
    print(f"\nMétriques sauvegardées dans {metrics_path}")

    # Afficher les scores du patient courant
    print("\nMétriques 3D du patient courant :")
    print(f"  Dice : {dice:.4f}")
    print(f"  Hausdorff : {hausdorff:.2f}")
    print(f"  ASSD : {assd_score:.2f}")
