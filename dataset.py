"""
dataset.py
PyTorch Dataset for pedestrian intention prediction.
Loads precomputed .npy skeleton and headpose files.
Returns one sample at a time to the training loop.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


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

        # Pre-filter samples that have .npy files on disk
        self.valid_samples = []
        for s in samples:
            dataset    = s['dataset']          # 'JAAD' or 'PIE'
            vid        = s['video_id']
            pid        = s['ped_id'].replace('/', '-')
            sf         = s['start_frame']
            stem       = f"{vid}_{pid}_{sf}"
            skel_path  = os.path.join(npy_root, dataset, stem + '_skeleton.npy')
            head_path  = os.path.join(npy_root, dataset, stem + '_headpose.npy')

            if os.path.exists(skel_path) and os.path.exists(head_path):
                self.valid_samples.append({
                    **s,
                    'skel_path': skel_path,
                    'head_path': head_path,
                })

        print(f"Dataset: {len(self.valid_samples)} valid samples "
              f"(filtered from {len(samples)} total)")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]

        # ── Load .npy files ───────────────────────────────────────────────────
        skeleton = np.load(sample['skel_path']).astype(np.float32)  # [16, 17, 4]
        headpose = np.load(sample['head_path']).astype(np.float32)  # [16, 3]

        # ── Augmentation (training only) ──────────────────────────────────────
        if self.augment:
            # Add small Gaussian noise to skeleton coordinates
            noise = np.random.normal(0, 0.01, skeleton.shape).astype(np.float32)
            skeleton = skeleton + noise
            # Randomly flip horizontally (mirror pedestrian)
            if np.random.rand() > 0.5:
                skeleton[:, :, 0] = 1.0 - skeleton[:, :, 0]  # flip x coords
                headpose[:, 0]    = -headpose[:, 0]            # flip yaw

        # ── Labels ────────────────────────────────────────────────────────────
        crossing_labels = torch.tensor(
            sample['crossing_labels'], dtype=torch.float32
        )  # [4] binary labels at 0.5s, 1s, 2s, 4s

        # Distraction label: majority vote over 16 frames
        distraction_seq = sample['distraction_seq']
        distraction_label = torch.tensor(
            1 if sum(distraction_seq) > len(distraction_seq) / 2 else 0,
            dtype=torch.long
        )  # scalar: 0=attentive, 1=distracted

        # Dataset flag: 0=JAAD, 1=PIE, 2=TITAN
        dataset_map  = {'JAAD': 0, 'PIE': 1, 'TITAN': 2}
        dataset_flag = torch.tensor(
            dataset_map.get(sample['dataset'], 0), dtype=torch.long
        )

        # Has distraction label: only JAAD and PIE have looking annotations
        has_distraction = torch.tensor(
            1 if sample['dataset'] in ['JAAD', 'PIE'] else 0,
            dtype=torch.float32
        )

        return {
            'skeleton':          torch.tensor(skeleton),        # [16, 17, 4]
            'headpose':          torch.tensor(headpose),        # [16, 3]
            'crossing_labels':   crossing_labels,               # [4]
            'distraction_label': distraction_label,             # scalar
            'dataset_flag':      dataset_flag,                  # scalar
            'has_distraction':   has_distraction,               # scalar
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

    # Split by video_id to avoid leakage
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

    # Test loading one sample
    sample = train_ds[0]
    print("\nSample keys:", list(sample.keys()))
    print("Skeleton shape:", sample['skeleton'].shape)
    print("Headpose shape:", sample['headpose'].shape)
    print("Crossing labels:", sample['crossing_labels'])
    print("Distraction label:", sample['distraction_label'])
    print("Dataset flag:", sample['dataset_flag'])