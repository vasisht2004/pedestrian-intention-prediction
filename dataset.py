"""
dataset.py
PyTorch Dataset for pedestrian intention prediction.
Loads precomputed .npy skeleton and headpose files.

Includes:
- Skeleton quality filter (removes MediaPipe failures)
- Per-channel standardization (mean=0, std=1) computed from training set
  This is critical because position channels (x_norm, y_norm) carry the
  discriminative signal (crossers are positioned differently) but get
  drowned by velocity noise without standardization.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Minimum mean absolute skeleton value to keep a sample
SKELETON_QUALITY_THRESHOLD = 0.05

# Per-channel normalization constants computed from training set
# Channels: [x_norm, y_norm, vx, vy]
SKEL_MEAN = np.array([0.5001, 0.3396, 0.0033, 0.0042], dtype=np.float32)
SKEL_STD  = np.array([0.3557, 0.3594, 0.2108, 0.2014], dtype=np.float32)


class PedestrianDataset(Dataset):
    def __init__(self, samples, npy_root, augment=False):
        """
        samples:  list of dicts from jaad_parser or pie_parser
        npy_root: root folder containing JAAD/ and PIE/ subfolders
        augment:  if True, apply noise augmentation (training only)
        """
        self.samples  = samples
        self.npy_root = npy_root
        self.augment  = augment

        # Pre-filter samples that have .npy files on disk AND pass quality check
        self.valid_samples = []
        n_total    = 0
        n_no_file  = 0
        n_low_qual = 0

        for s in samples:
            dataset    = s['dataset']
            vid        = s['video_id']
            pid        = s['ped_id'].replace('/', '-')
            sf         = s['start_frame']
            stem       = f"{vid}_{pid}_{sf}"
            skel_path  = os.path.join(npy_root, dataset, stem + '_skeleton.npy')
            head_path  = os.path.join(npy_root, dataset, stem + '_headpose.npy')
            n_total   += 1

            if not (os.path.exists(skel_path) and os.path.exists(head_path)):
                n_no_file += 1
                continue

            # Skeleton quality check
            skel = np.load(skel_path).astype(np.float32)
            if np.abs(skel).mean() < SKELETON_QUALITY_THRESHOLD:
                n_low_qual += 1
                continue

            self.valid_samples.append({
                **s,
                'skel_path': skel_path,
                'head_path': head_path,
            })

        print(f"Dataset: {len(self.valid_samples)} valid samples "
              f"(from {n_total} total, {n_no_file} missing files, "
              f"{n_low_qual} low-quality skeletons filtered)")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]

        # ── Load .npy files ───────────────────────────────────────────────────
        skeleton = np.load(sample['skel_path']).astype(np.float32)  # [16, 17, 4]
        headpose = np.load(sample['head_path']).astype(np.float32)  # [16, 3]

        # ── Per-channel standardization ───────────────────────────────────────
        # skeleton shape: [16, 17, 4] — last dim is [x_norm, y_norm, vx, vy]
        # Standardize each channel to mean=0, std=1
        skeleton = (skeleton - SKEL_MEAN) / (SKEL_STD + 1e-8)

        # ── Augmentation (training only) ──────────────────────────────────────
        if self.augment:
            # Add small Gaussian noise (in standardized space)
            noise = np.random.normal(0, 0.05, skeleton.shape).astype(np.float32)
            skeleton = skeleton + noise
            # Randomly flip horizontally (mirror pedestrian)
            if np.random.rand() > 0.5:
                skeleton[:, :, 0] = -skeleton[:, :, 0]   # flip x (already centered at 0)
                skeleton[:, :, 2] = -skeleton[:, :, 2]   # flip vx
                headpose[:, 0]    = -headpose[:, 0]       # flip yaw

        # ── Labels ────────────────────────────────────────────────────────────
        crossing_labels = torch.tensor(
            sample['crossing_labels'], dtype=torch.float32
        )

        distraction_seq = sample['distraction_seq']
        distraction_label = torch.tensor(
            1 if sum(distraction_seq) > len(distraction_seq) / 2 else 0,
            dtype=torch.long
        )

        dataset_map  = {'JAAD': 0, 'PIE': 1, 'TITAN': 2}
        dataset_flag = torch.tensor(
            dataset_map.get(sample['dataset'], 0), dtype=torch.long
        )

        has_distraction = torch.tensor(
            1 if sample['dataset'] in ['JAAD', 'PIE'] else 0,
            dtype=torch.float32
        )

        return {
            'skeleton':          torch.tensor(skeleton),
            'headpose':          torch.tensor(headpose),
            'crossing_labels':   crossing_labels,
            'distraction_label': distraction_label,
            'dataset_flag':      dataset_flag,
            'has_distraction':   has_distraction,
        }


def build_datasets(jaad_root, pie_annot_root, npy_root, val_split=0.15, test_split=0.15):
    """
    Builds train/val/test splits from JAAD and PIE.
    Split is done by video_id to prevent data leakage.
    """
    from jaad_parser import parse_jaad_dataset
    from pie_parser  import parse_pie_dataset

    print("Parsing JAAD...")
    jaad_samples = parse_jaad_dataset(jaad_root)

    print("Parsing PIE...")
    pie_samples  = parse_pie_dataset(pie_annot_root)

    all_samples  = jaad_samples + pie_samples
    print(f"Total samples: {len(all_samples)}")

    video_ids = sorted(set(s['video_id'] for s in all_samples))
    np.random.seed(42)
    np.random.shuffle(video_ids)

    n          = len(video_ids)
    n_test     = int(n * test_split)
    n_val      = int(n * val_split)
    test_vids  = set(video_ids[:n_test])
    val_vids   = set(video_ids[n_test:n_test + n_val])
    train_vids = set(video_ids[n_test + n_val:])

    train_samples = [s for s in all_samples if s['video_id'] in train_vids]
    val_samples   = [s for s in all_samples if s['video_id'] in val_vids]
    test_samples  = [s for s in all_samples if s['video_id'] in test_vids]

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_ds = PedestrianDataset(train_samples, npy_root, augment=True)
    val_ds   = PedestrianDataset(val_samples,   npy_root, augment=False)
    test_ds  = PedestrianDataset(test_samples,  npy_root, augment=False)

    return train_ds, val_ds, test_ds


if __name__ == '__main__':
    train_ds, val_ds, test_ds = build_datasets(
        jaad_root      = '/Users/payas/JAAD',
        pie_annot_root = '/Users/payas/PIE_annotations/annotations/annotations',
        npy_root       = '/Users/payas/pedestrian_project/npy_data',
    )

    sample = train_ds[0]
    print("\nSample keys:", list(sample.keys()))
    print("Skeleton shape:", sample['skeleton'].shape)
    print("Skeleton stats (should be ~mean=0, std=1):")
    sk = sample['skeleton'].numpy()
    for ch, name in enumerate(['x_norm', 'y_norm', 'vx', 'vy']):
        print(f"  {name}: mean={sk[:,:,ch].mean():.3f} std={sk[:,:,ch].std():.3f}")