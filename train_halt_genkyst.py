import os
import torch
import argparse
import random
import numpy as np
import glob
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split

from nets.whichnet import whichnet
from utils.train_utils import launch_training
from utils.sampler import TwoStreamBatchSampler
from datasets.dataset_semi_supervised import tiny_dataset

def worker_init_fn(worker_id):
    random.seed(1337 + worker_id)

def semi_supervised_collate(batch):
    images, masks = [], []
    for item in batch:
        if isinstance(item, tuple):
            img, mask = item
            images.append(img)
            masks.append(mask)
        else:
            images.append(item)
            masks.append(None)
    return images, masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lambda_consistency', type=float, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_test/')
    parser.add_argument('--vgg', action='store_true')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    cudnn.benchmark = True

    # === Chargement dynamique des patients labellisés et non-labellisés ===
    base_dir = "/home/hemery/code_halt_semi_supervised/data_halt_genkyst/train"
    labeled_img_dir = os.path.join(base_dir, "labeled/T2")
    labeled_mask_dir = os.path.join(base_dir, "labeled/mask")
    unlabeled_dir = os.path.join(base_dir, "unlabeled")

    labeled_ids = []
    labeled_series = []

    for img_path in sorted(glob.glob(os.path.join(labeled_img_dir, "*.nii.gz"))):
        filename = os.path.basename(img_path)
        try:
            patient_id, series, _ = filename.replace(".nii.gz", "").split("-")
        except ValueError:
            print(f"Skipping malformed labeled file: {filename}")
            continue

        mask_l = os.path.join(labeled_mask_dir, f"{patient_id}-{series}-LK.nii.gz")
        mask_r = os.path.join(labeled_mask_dir, f"{patient_id}-{series}-RK.nii.gz")

        if os.path.exists(mask_l) and os.path.exists(mask_r):
            labeled_ids.append(patient_id)
            labeled_series.append(int(series))

    print(f"✓ Found {len(labeled_ids)} labeled patients")

    # === Séparation train/val sur les données labellisées ===
    train_ids, val_ids, train_series, val_series = train_test_split(
        labeled_ids, labeled_series, test_size=8, random_state=42
    )

    # === Chargement des patients non-labellisés ===
    unlabeled_ids = []
    unlabeled_series = []

    for img_path in sorted(glob.glob(os.path.join(unlabeled_dir, "*.nii.gz"))):
        filename = os.path.basename(img_path)
        try:
            patient_id, series, _ = filename.replace(".nii.gz", "").split("-")
        except ValueError:
            print(f"Skipping malformed unlabeled file: {filename}")
            continue

        unlabeled_ids.append(patient_id)
        unlabeled_series.append(int(series))

    print(f"✓ Found {len(unlabeled_ids)} unlabeled patients")

    # === Création des datasets ===
    train_flags = [True] * len(train_ids) + [False] * len(unlabeled_ids)
    val_flags = [True] * len(val_ids)

    train_dataset = tiny_dataset(
        ids=train_ids + unlabeled_ids,
        series=train_series + unlabeled_series,
        labeled_flags=train_flags,
        size=args.size,
        vgg=args.vgg,
        base_dir=base_dir
    )

    val_dataset = tiny_dataset(
        ids=val_ids,
        series=val_series,
        labeled_flags=val_flags,
        size=args.size,
        vgg=args.vgg,
        base_dir=base_dir
    )

    # === Indices labellés / non-labellés pour le sampler ===
    train_labeled_idxs = [i for i, entry in enumerate(train_dataset.entries) if entry['labeled']]
    train_unlabeled_idxs = [i for i, entry in enumerate(train_dataset.entries) if not entry['labeled']]

    print(f"✓ Training set: {len(train_labeled_idxs)} labeled, {len(train_unlabeled_idxs)} unlabeled")
    print(f"✓ Validation set: {len(val_dataset)} labeled")

    # === Sampler & DataLoaders ===
    train_sampler = TwoStreamBatchSampler(
        primary_indices=train_labeled_idxs,
        secondary_indices=train_unlabeled_idxs,
        batch_size=args.batch_size,
        secondary_batch_size=args.batch_size // 2
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=semi_supervised_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # === Modèle ===
    net, _ = whichnet(net_id=args.model, n_classes=1, img_size=args.size)
    net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss() # appeler en argument pour minimiser la BCE pour calcul loss supervisée

    launch_training(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        save_dir=args.save_dir,
        lambda_consistency=args.lambda_consistency,
        enable_plot=False  # <- désactive plot
    )
