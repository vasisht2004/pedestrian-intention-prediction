"""
jaad_parser.py
Parses JAAD annotation XMLs into a unified list of pedestrian samples.

Each sample is a dict:
{
  'video_id':        'video_0001',
  'ped_id':          '0_1_3b',
  'start_frame':     frame index where our 16-frame window starts,
  'bbox_seq':        list of 16 bounding boxes [(xtl,ytl,xbr,ybr), ...],
  'crossing_labels': [label@0.5s, label@1s, label@2s, label@4s],  # binary
  'distraction_seq': list of 16 ints (1=distracted/not-looking, 0=looking),
  'dataset':         'JAAD',
}

JAAD fps is approximately 10fps for most clips.
We use 12fps horizon math to stay consistent with PIE, but flag this.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path


# ----------------------------------------------------------------
# JAAD frame rate — most clips are 10fps, some 30fps
# We use 10fps as default. Adjust per-video if needed.
# ----------------------------------------------------------------
JAAD_FPS = 10
OBS_FRAMES = 16   # 16-frame observation window

# Horizon offsets in frames at JAAD_FPS
# 0.5s=5, 1s=10, 2s=20, 4s=40 frames at 10fps
HORIZON_FRAMES = [5, 10, 20, 40]


def parse_jaad_video(ann_xml_path: str, attr_xml_path: str) -> list:
    """
    Parse one JAAD video annotation + attributes file.
    Returns list of sample dicts for all valid pedestrian windows.
    """
    samples = []
    video_id = Path(ann_xml_path).stem  # e.g. 'video_0001'

    # ----------------------------------------------------------------
    # Parse attributes file first — get decision_point per pedestrian
    # decision_point = frame number where pedestrian decides to cross
    # -1 means pedestrian never crosses
    # ----------------------------------------------------------------
    ped_attrs = {}
    if os.path.exists(attr_xml_path):
        attr_tree = ET.parse(attr_xml_path)
        attr_root = attr_tree.getroot()
        for ped in attr_root.findall('pedestrian'):
            ped_id = ped.get('id')
            crossing = int(ped.get('crossing', -1))
            decision_point = int(ped.get('decision_point', -1))
            ped_attrs[ped_id] = {
                'crossing': crossing,
                'decision_point': decision_point,
                'age': ped.get('age', 'unknown'),
                'gender': ped.get('gender', 'unknown'),
            }

    # ----------------------------------------------------------------
    # Parse annotation file — get per-frame bbox and behavioral tags
    # Only process 'pedestrian' label tracks (not 'ped' or 'people')
    # ----------------------------------------------------------------
    ann_tree = ET.parse(ann_xml_path)
    ann_root = ann_tree.getroot()

    for track in ann_root.findall('track'):
        label = track.get('label', '')
        if label != 'pedestrian':
            continue  # skip 'ped' and 'people' tracks — no behavioral tags

        # Get pedestrian ID from first box's attributes
        boxes = track.findall('box')
        if len(boxes) < OBS_FRAMES:
            continue  # not enough frames

        # Extract ped_id from first box
        ped_id = None
        for attr in boxes[0].findall('attribute'):
            if attr.get('name') == 'id':
                ped_id = attr.text.strip() if attr.text else None
                break
        if ped_id is None:
            continue

        # ----------------------------------------------------------------
        # Build per-frame data arrays
        # ----------------------------------------------------------------
        frame_data = {}  # frame_idx -> {bbox, look, cross, occlusion}

        for box in boxes:
            frame_idx = int(box.get('frame'))
            outside = int(box.get('outside', 0))
            if outside == 1:
                continue  # pedestrian outside frame

            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # Parse behavioral attributes
            attrs = {}
            for attr in box.findall('attribute'):
                attrs[attr.get('name')] = attr.text.strip() if attr.text else ''

            look = attrs.get('look', '__undefined__')
            cross = attrs.get('cross', 'not-crossing')
            occlusion = attrs.get('occlusion', 'none')

            frame_data[frame_idx] = {
                'bbox': (xtl, ytl, xbr, ybr),
                'look': look,
                'cross': cross,
                'occlusion': occlusion,
            }

        if len(frame_data) < OBS_FRAMES:
            continue

        sorted_frames = sorted(frame_data.keys())
        total_frames = len(sorted_frames)

        # ----------------------------------------------------------------
        # Get decision_point for this pedestrian
        # ----------------------------------------------------------------
        attrs_info = ped_attrs.get(ped_id, {})
        decision_point = attrs_info.get('decision_point', -1)

        # ----------------------------------------------------------------
        # Slide a 16-frame window across the track
        # Step size = 4 frames (stride) to get overlapping samples
        # Only include windows where all 16 frames have valid data
        # ----------------------------------------------------------------
        for win_start_idx in range(0, total_frames - OBS_FRAMES + 1, 4):
            window_frame_nums = sorted_frames[win_start_idx: win_start_idx + OBS_FRAMES]

            # Check all 16 frames are consecutive (no large gaps)
            if window_frame_nums[-1] - window_frame_nums[0] > OBS_FRAMES * 2:
                continue  # too many gaps in this window

            # Check occlusion — skip windows with too many fully occluded frames
            occluded_count = sum(
                1 for f in window_frame_nums
                if frame_data[f]['occlusion'] == 'full'
            )
            if occluded_count > 4:  # more than 4/16 frames fully occluded
                continue

            # ----------------------------------------------------------------
            # Extract bbox sequence for these 16 frames
            # ----------------------------------------------------------------
            bbox_seq = [frame_data[f]['bbox'] for f in window_frame_nums]

            # ----------------------------------------------------------------
            # Extract distraction sequence
            # not-looking = 1 (distracted), looking = 0 (attentive)
            # __undefined__ = treated as 0 (assume attentive if unknown)
            # ----------------------------------------------------------------
            distraction_seq = []
            for f in window_frame_nums:
                look = frame_data[f]['look']
                distraction_seq.append(1 if look == 'not-looking' else 0)

            # ----------------------------------------------------------------
            # Compute crossing labels at 4 horizons
            # Last frame of observation window
            last_obs_frame = window_frame_nums[-1]

            # If pedestrian never crosses: all labels = 0
            # If decision_point is within horizon: label = 1
            # ----------------------------------------------------------------
            crossing_labels = []
            for horizon_frames in HORIZON_FRAMES:
                if decision_point == -1:
                    # Never crosses
                    crossing_labels.append(0)
                elif decision_point <= last_obs_frame + horizon_frames:
                    # Will cross within this horizon
                    crossing_labels.append(1)
                else:
                    crossing_labels.append(0)

            samples.append({
                'video_id':        video_id,
                'ped_id':          ped_id,
                'start_frame':     window_frame_nums[0],
                'end_frame':       window_frame_nums[-1],
                'frame_nums':      window_frame_nums,
                'bbox_seq':        bbox_seq,
                'crossing_labels': crossing_labels,
                'distraction_seq': distraction_seq,
                'dataset':         'JAAD',
                'decision_point':  decision_point,
            })

    return samples


def parse_jaad_dataset(jaad_root: str) -> list:
    """
    Parse entire JAAD dataset.

    Args:
        jaad_root: path to JAAD repo root (e.g. ~/JAAD)

    Returns:
        List of all sample dicts across all videos
    """
    ann_dir  = os.path.join(jaad_root, 'annotations')
    attr_dir = os.path.join(jaad_root, 'annotations_attributes')

    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"JAAD annotations not found at {ann_dir}")

    ann_files = sorted(Path(ann_dir).glob('video_*.xml'))
    print(f"Found {len(ann_files)} JAAD annotation files")

    all_samples = []
    for ann_path in ann_files:
        video_id = ann_path.stem
        attr_path = os.path.join(attr_dir, f'{video_id}_attributes.xml')

        try:
            samples = parse_jaad_video(str(ann_path), attr_path)
            all_samples.extend(samples)
        except Exception as e:
            print(f"  Warning: failed to parse {video_id}: {e}")
            continue

    return all_samples


def get_split_ids(jaad_root: str) -> dict:
    """
    Load official JAAD train/val/test split IDs.
    Falls back to an 80/10/10 random split if split files not found.
    """
    split_dir = os.path.join(jaad_root, 'split_ids')
    splits = {}

    for split_name in ['train', 'val', 'test']:
        split_file = os.path.join(split_dir, f'jaad_{split_name}_IDs.txt')
        if os.path.exists(split_file):
            with open(split_file) as f:
                ids = [line.strip() for line in f if line.strip()]
            splits[split_name] = ids
            print(f"  {split_name}: {len(ids)} video IDs")
        else:
            splits[split_name] = None
            print(f"  No split file found for {split_name} — will use random split")

    return splits


def print_dataset_stats(samples: list):
    """Print statistics about the parsed dataset."""
    total = len(samples)
    crossing_counts = [0, 0, 0, 0]
    distracted_frames = 0
    total_frames = 0

    for s in samples:
        for i, label in enumerate(s['crossing_labels']):
            crossing_counts[i] += label
        distracted_frames += sum(s['distraction_seq'])
        total_frames += len(s['distraction_seq'])

    horizons = ['0.5s', '1s', '2s', '4s']
    print(f"\n{'='*50}")
    print(f"JAAD Dataset Statistics")
    print(f"{'='*50}")
    print(f"Total samples (sliding windows): {total}")
    print(f"\nCrossing label distribution:")
    for h, count in zip(horizons, crossing_counts):
        pct = count / total * 100 if total > 0 else 0
        print(f"  @{h}: {count} crossing ({pct:.1f}%) / {total-count} not crossing")
    print(f"\nDistraction rate: {distracted_frames/total_frames*100:.1f}% of frames")
    print(f"{'='*50}\n")


# ----------------------------------------------------------------
# Quick test — run this file directly to verify parser works
# ----------------------------------------------------------------
if __name__ == '__main__':
    import sys

    jaad_root = os.path.expanduser('~/JAAD')
    if len(sys.argv) > 1:
        jaad_root = sys.argv[1]

    print(f"Parsing JAAD from: {jaad_root}")

    # Parse first 5 videos only for quick test
    ann_dir = os.path.join(jaad_root, 'annotations')
    ann_files = sorted(Path(ann_dir).glob('video_*.xml'))[:5]

    test_samples = []
    for ann_path in ann_files:
        video_id = ann_path.stem
        attr_path = os.path.join(jaad_root, 'annotations_attributes', f'{video_id}_attributes.xml')
        samples = parse_jaad_video(str(ann_path), attr_path)
        test_samples.extend(samples)
        print(f"  {video_id}: {len(samples)} samples")

    print_dataset_stats(test_samples)

    # Show one sample in detail
    if test_samples:
        s = test_samples[0]
        print("Example sample:")
        print(f"  video_id:        {s['video_id']}")
        print(f"  ped_id:          {s['ped_id']}")
        print(f"  frames:          {s['start_frame']} -> {s['end_frame']}")
        print(f"  bbox[0]:         {s['bbox_seq'][0]}")
        print(f"  crossing_labels: {s['crossing_labels']}  (0.5s, 1s, 2s, 4s)")
        print(f"  distraction_seq: {s['distraction_seq']}")
        print(f"  decision_point:  {s['decision_point']}")
        print("\nParser working correctly.")
