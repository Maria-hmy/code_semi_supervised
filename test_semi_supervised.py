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
    parser = argparse.ArgumentParser(description='Segmentation rénale semi_supervisée T2 HALT')
    parser.add_argument('-p', '--patient', type=str, required=True, help='ID patient (ex: B9309076)')
    parser.add_argument('-s', '--serie', type=int, required=True, help='ID série (ex: 2)')
    parser.add_argument('-o', '--output', type=str, default='./outputs/', help='Dossier de sortie')
    parser.add_argument('--gt_dir', type=str, default='./GT', help="Dossier contenant les GT (masques)")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    net_id = 1
    n_classes = 1
    size = 256
    modality = 'T2'


    net, vgg = whichnet(net_id, n_classes, size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # Chargement du modèle
    state_dict = torch.load('best_model_teacher_C4_8_64_F0.pth', map_location=device)
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    print("Paramètres manquants :", missing)
    print("Paramètres inattendus :", unexpected)


    
    net.eval()
    logging.info(f"Modèle chargé sur {device}")



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

    with torch.no_grad():
        t2_data = dataset.exam.T2.get_fdata()
        depth = t2_data.shape[1]

        for idx, data in enumerate(loader):
            image = data.to(device=device, dtype=torch.float32)
            pred = net(image)


            prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            #print(f"Prediction sigmoid - min: {prob.min()}, max: {prob.max()}, mean: {prob.mean()}")

        
            target_shape = (t2_data.shape[2], t2_data.shape[0])  # inversion pour rot90 ############################################
            prob_resized = resize(
                prob,
                target_shape,
                order=1,
                preserve_range=True,
                anti_aliasing=True
            )

            array[:, idx, :] = np.rot90(prob_resized, -1)[::-1, :] > 0.5  # seuil 

    nib.save(nib.Nifti1Image(array.astype(np.uint16), affine, header),
             os.path.join(args.output, f"{args.patient}-{args.serie:02d}-prediction.nii.gz"))
    
   

    t2_output_path = os.path.join(args.output, f"{args.patient}-{args.serie:02d}-T2.nii.gz")
    nib.save(nib.Nifti1Image(t2_data, affine, header), t2_output_path)

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

        print("Forme de la prédiction :", array.shape)
        print("Forme de la GT combinée :", gt_array.shape)
        print("Nombre de voxels prédits positifs :", np.sum(array))
        print("Nombre de voxels GT positifs :", np.sum(gt_array))

        if gt_array.shape != array.shape:
            logging.warning(f"Shape mismatch: prédiction {array.shape} vs GT {gt_array.shape}")
            dice, hausdorff, assd_score = -1, -1, -1
        elif np.sum(array) == 0 or np.sum(gt_array) == 0:
            logging.warning("GT ou prédiction vide (aucun voxel positif)")
            dice, hausdorff, assd_score = -1, -1, -1
        else:
            try:
                
                pred_bin = array.astype(np.uint8) # Sans get2largest

                #garder seulement les 2 plus grandes régions connectées
                #filtered_array = get2LargestConnectedAreas(array.astype(np.uint8))
                #pred_bin = filtered_array.astype(np.uint8)

                gt_bin = gt_array.astype(np.uint8)
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
        
        
        #nib.save(nib.Nifti1Image(filtered_array.astype(np.uint16), affine, header),
                 #os.path.join(args.output, f"{args.patient}-{args.serie:02d}-prediction_filtered.nii.gz"))

    metrics_path = os.path.join(args.output, "metrics_summary.csv")
    results = {
        "Patient": args.patient,
        "Serie": args.serie,
        "Dice": dice,
        "HD": hausdorff,
        "ASSD": assd_score,
    }

    if os.path.exists(metrics_path):
        df_existing = pd.read_csv(metrics_path)
        df_existing = df_existing[df_existing["Patient"] != "MOYENNE"]
    else:
        df_existing = pd.DataFrame()

    df_current = pd.DataFrame([results])
    df_all = pd.concat([df_existing, df_current], ignore_index=True)

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

    df_all.to_csv(metrics_path, index=False)
    print(f"\nMétriques sauvegardées dans {metrics_path}")

    print("\nMétriques 3D du patient courant :")
    print(f"  Dice : {dice:.4f}")
    print(f"  Hausdorff : {hausdorff:.2f}")
    print(f"  ASSD : {assd_score:.2f}")
