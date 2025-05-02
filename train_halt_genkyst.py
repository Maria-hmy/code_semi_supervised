import os
import torch
import argparse
import random
import numpy as np
import glob
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
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
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--semi_weight', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/')
    parser.add_argument('--vgg', action='store_true')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    cudnn.benchmark = True

    # === Chargement dynamique des patients labellisés et non-labellisés ===
    base_dir = "/home/hemery/code_halt_semi_supervised/data_halt_genkyst"
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

    # === Créer les flags labellisés / non labellisés ===
    labeled_flags = [True] * len(labeled_ids) + [False] * len(unlabeled_ids)

    # === Créer le dataset dynamique ===
    dataset = tiny_dataset(
        ids=labeled_ids + unlabeled_ids,
        series=labeled_series + unlabeled_series,
        labeled_flags=labeled_flags,
        size=args.size,
        vgg=args.vgg,
        base_dir=base_dir
    )

    # === Indices labellisés / non-labellisés ===
    labeled_idxs = [i for i, entry in enumerate(dataset.entries) if entry['labeled']]
    unlabeled_idxs = [i for i, entry in enumerate(dataset.entries) if not entry['labeled']]

    print(f"Labeled slices: {len(labeled_idxs)}")
    print(f"Unlabeled slices: {len(unlabeled_idxs)}")
    print(f" Total slices: {len(dataset)}")

    # === Sampler personnalisé ===
    batch_sampler = TwoStreamBatchSampler(
        primary_indices=labeled_idxs,
        secondary_indices=unlabeled_idxs,
        batch_size=args.batch_size,
        secondary_batch_size=args.batch_size // 2
    )

    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=semi_supervised_collate
    )

    net, _ = whichnet(net_id=args.model, n_classes=1, img_size=args.size)
    net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    launch_training(
        model=net,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        save_dir=args.save_dir,
        semi_weight=args.semi_weight,
        enable_plot=False  # <- désactive plot 
)

