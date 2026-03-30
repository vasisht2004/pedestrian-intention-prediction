"""
pie_parser.py
Parses PIE annotation XMLs into a unified list of pedestrian samples.
Each sample matches the same format as jaad_parser.py for compatibility.
PIE fps: 30fps. We sample every 2-3 frames to get ~12fps effective rate.
Observation window: 16 frames. Prediction horizons: 0.5s, 1s, 2s, 4s.
At 30fps: 0.5s=15f, 1s=30f, 2s=60f, 4s=120f
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict

PIE_FPS        = 30
OBS_LEN        = 16
FRAME_STEP     = 15  # sample every 2 frames → effective 15fps ≈ 12fps target
HORIZONS_SEC   = [0.5, 1.0, 2.0, 4.0]
HORIZONS_FRAMES = [int(h * PIE_FPS) for h in HORIZONS_SEC]  # [15, 30, 60, 120]


def parse_pie_dataset(pie_annot_root: str) -> list:
    """
    pie_annot_root: path to folder containing set01/set02/.../set06
                    e.g. /Users/payas/PIE_annotations/annotations/annotations
    Returns list of sample dicts compatible with jaad_parser output.
    """
    samples = []

    set_dirs = sorted([
        d for d in os.listdir(pie_annot_root)
        if os.path.isdir(os.path.join(pie_annot_root, d)) and d.startswith('set')
    ])

    for set_id in set_dirs:
        set_path = os.path.join(pie_annot_root, set_id)
        xml_files = sorted([f for f in os.listdir(set_path) if f.endswith('_annt.xml')])

        for xml_file in xml_files:
            video_id = xml_file.replace('_annt.xml', '')  # e.g. video_0001
            xml_path = os.path.join(set_path, xml_file)

            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Group all tracks by pedestrian id
            ped_tracks = defaultdict(list)

            for track in root.findall('track'):
                if track.attrib.get('label') != 'pedestrian':
                    continue

                for box in track.findall('box'):
                    frame = int(box.attrib['frame'])
                    xtl   = float(box.attrib['xtl'])
                    ytl   = float(box.attrib['ytl'])
                    xbr   = float(box.attrib['xbr'])
                    ybr   = float(box.attrib['ybr'])

                    ped_id   = None
                    cross    = 'not-crossing'
                    look     = 'looking'

                    for attr in box.findall('attribute'):
                        name = attr.attrib['name']
                        val  = attr.text.strip() if attr.text else ''
                        if name == 'id':
                            ped_id = val
                        elif name == 'cross':
                            cross = val
                        elif name == 'look':
                            look = val

                    if ped_id is None:
                        continue

                    ped_tracks[ped_id].append({
                        'frame': frame,
                        'bbox':  (xtl, ytl, xbr, ybr),
                        'cross': cross,
                        'look':  look,
                    })

            # Build sliding window samples for each pedestrian
            for ped_id, frames_data in ped_tracks.items():
                frames_data.sort(key=lambda x: x['frame'])

                # Build frame-indexed lookup
                frame_lookup = {d['frame']: d for d in frames_data}
                frame_nums   = sorted(frame_lookup.keys())

                if len(frame_nums) < OBS_LEN + max(HORIZONS_FRAMES):
                    continue

                # Sliding window with step = FRAME_STEP
                i = 0
                while i + OBS_LEN * FRAME_STEP + max(HORIZONS_FRAMES) <= len(frame_nums):
                    # Pick 16 frames spaced FRAME_STEP apart
                    obs_indices = frame_nums[i: i + OBS_LEN * FRAME_STEP: FRAME_STEP]

                    if len(obs_indices) < OBS_LEN:
                        i += FRAME_STEP
                        continue

                    last_obs_frame = obs_indices[-1]

                    # Find decision point: first frame where cross changes to 'crossing'
                    future_frames = [
                        fn for fn in frame_nums if fn > last_obs_frame
                    ]

                    decision_point = None
                    for fn in future_frames:
                        if frame_lookup[fn]['cross'] == 'crossing':
                            decision_point = fn
                            break

                    # Build crossing labels at 4 horizons
                    crossing_labels = []
                    for horizon_f in HORIZONS_FRAMES:
                        target_frame = last_obs_frame + horizon_f
                        # Check if crossing within this horizon
                        crossed = 0
                        for fn in future_frames:
                            if fn <= target_frame and frame_lookup[fn]['cross'] == 'crossing':
                                crossed = 1
                                break
                        crossing_labels.append(crossed)

                    # Build bbox sequence and distraction sequence
                    bbox_seq       = [frame_lookup[fn]['bbox'] for fn in obs_indices]
                    distraction_seq = [
                        1 if frame_lookup[fn]['look'] == 'not-looking' else 0
                        for fn in obs_indices
                    ]

                    samples.append({
                        'video_id':         f'{set_id}_{video_id}',
                        'ped_id':           ped_id,
                        'start_frame':      obs_indices[0],
                        'end_frame':        obs_indices[-1],
                        'frame_nums':       obs_indices,
                        'bbox_seq':         bbox_seq,
                        'crossing_labels':  crossing_labels,
                        'distraction_seq':  distraction_seq,
                        'dataset':          'PIE',
                        'decision_point':   decision_point,
                        'set_id':           set_id,
                    })

                    i += FRAME_STEP

    return samples


if __name__ == '__main__':
    PIE_ANNOT_ROOT = '/Users/payas/PIE_annotations/annotations/annotations'
    samples = parse_pie_dataset(PIE_ANNOT_ROOT)
    print(f'Total PIE samples: {len(samples)}')

    crossing = sum(1 for s in samples if s['crossing_labels'][2] == 1)
    print(f'Crossing @2s: {crossing} ({100*crossing/len(samples):.1f}%)')
    print(f'Distraction rate: {sum(sum(s["distraction_seq"]) for s in samples) / (len(samples)*16) * 100:.1f}%')
    print(f'Example sample: {samples[0]}')