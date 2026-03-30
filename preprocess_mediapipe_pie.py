import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import sys
import multiprocessing as multiproc
from functools import partial

sys.path.append('/Users/payas/pedestrian_project')
from pie_parser import parse_pie_dataset

PIE_CLIPS_DIR = '/Users/payas/Downloads/PIE_clips'
OUTPUT_DIR    = '/Users/payas/pedestrian_project/npy_data/PIE'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0   ],
    [0.0,   -63.6,  -12.5 ],
    [-43.3,  32.7,  -26.0 ],
    [43.3,   32.7,  -26.0 ],
    [-28.9, -28.9,  -24.1 ],
    [28.9,  -28.9,  -24.1 ],
], dtype=np.float64)

FACE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]


def get_head_pose(landmarks_2d, frame_w, frame_h):
    focal   = frame_w
    cam_mat = np.array([
        [focal, 0,     frame_w / 2],
        [0,     focal, frame_h / 2],
        [0,     0,     1          ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    success, rvec, _ = cv2.solvePnP(
        FACE_3D_MODEL, landmarks_2d, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return np.zeros(3)
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-rmat[2,0], sy))
        yaw   = np.degrees(np.arctan2( rmat[2,1], rmat[2,2]))
        roll  = np.degrees(np.arctan2( rmat[1,0], rmat[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[2,0], sy))
        yaw   = 0.0
        roll  = np.degrees(np.arctan2(-rmat[0,1], rmat[1,1]))
    return np.array([yaw, pitch, roll], dtype=np.float32)


def process_chunk(samples_chunk, worker_id):
    """Each worker process runs its own MediaPipe instance."""
    mp_pose      = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    saved   = 0
    skipped = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose_model, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_model:

        for sample in samples_chunk:
            set_id   = sample['set_id']
            vid_name = sample['video_id'].replace(f'{set_id}_', '')
            video_path = os.path.join(PIE_CLIPS_DIR, set_id, vid_name + '.mp4')

            vid  = sample['video_id']
            pid  = sample['ped_id'].replace('/', '-')
            sf   = sample['start_frame']
            stem = f"{vid}_{pid}_{sf}"

            skel_path = os.path.join(OUTPUT_DIR, stem + '_skeleton.npy')
            head_path = os.path.join(OUTPUT_DIR, stem + '_headpose.npy')

            if os.path.exists(skel_path) and os.path.exists(head_path):
                saved += 1
                continue

            if not os.path.exists(video_path):
                skipped += 1
                continue

            cap          = cv2.VideoCapture(video_path)
            skeleton_seq = np.zeros((16, 17, 4), dtype=np.float32)
            headpose_seq = np.zeros((16, 3),     dtype=np.float32)
            prev_joints  = None

            for t, (frame_num, bbox) in enumerate(zip(sample['frame_nums'], sample['bbox_seq'])):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    if prev_joints is not None:
                        skeleton_seq[t, :, :2] = prev_joints[:, :2]
                    continue

                frame_h, frame_w = frame.shape[:2]
                xtl, ytl, xbr, ybr = [int(v) for v in bbox]
                xtl, ytl = max(0, xtl), max(0, ytl)
                xbr, ybr = min(frame_w, xbr), min(frame_h, ybr)
                crop = frame[ytl:ybr, xtl:xbr]
                if crop.size == 0:
                    continue

                bw       = xbr - xtl
                bh       = ybr - ytl
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                pose_result = pose_model.process(crop_rgb)
                joints      = np.zeros((17, 4), dtype=np.float32)

                if pose_result.pose_landmarks:
                    lms  = pose_result.pose_landmarks.landmark
                    keep = [0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]
                    for j, idx in enumerate(keep):
                        lm = lms[idx]
                        if lm.visibility > 0.5:
                            joints[j, 0] = lm.x
                            joints[j, 1] = lm.y
                        else:
                            if prev_joints is not None:
                                joints[j, 0] = prev_joints[j, 0]
                                joints[j, 1] = prev_joints[j, 1]

                if prev_joints is not None:
                    joints[:, 2] = joints[:, 0] - prev_joints[:, 0]
                    joints[:, 3] = joints[:, 1] - prev_joints[:, 1]

                skeleton_seq[t] = joints
                prev_joints     = joints.copy()

                face_result = face_model.process(crop_rgb)
                if face_result.multi_face_landmarks:
                    fl     = face_result.multi_face_landmarks[0].landmark
                    pts_2d = np.array([
                        [fl[i].x * bw, fl[i].y * bh] for i in FACE_LANDMARK_IDS
                    ], dtype=np.float64)
                    headpose_seq[t] = get_head_pose(pts_2d, bw, bh)
                else:
                    if t > 0:
                        headpose_seq[t] = headpose_seq[t-1]

            cap.release()
            np.save(skel_path, skeleton_seq)
            np.save(head_path, headpose_seq)
            saved += 1

    return saved, skipped


def main():
    print("Loading PIE samples...")
    PIE_ANNOT_ROOT = '/Users/payas/PIE_annotations/annotations/annotations'
    samples        = parse_pie_dataset(PIE_ANNOT_ROOT)
    print(f"Total samples: {len(samples)}")

    NUM_WORKERS = 6
    chunk_size  = len(samples) // NUM_WORKERS
    chunks      = [
        samples[i * chunk_size: (i + 1) * chunk_size]
        for i in range(NUM_WORKERS)
    ]
    # Add remainder to last chunk
    chunks[-1].extend(samples[NUM_WORKERS * chunk_size:])

    print(f"Splitting {len(samples)} samples across {NUM_WORKERS} workers...")

    with multiproc.Pool(processes=NUM_WORKERS) as pool:
        args    = [(chunk, i) for i, chunk in enumerate(chunks)]
        results = pool.starmap(process_chunk, args)

    total_saved   = sum(r[0] for r in results)
    total_skipped = sum(r[1] for r in results)
    print(f"\nDone. Saved: {total_saved}  Skipped: {total_skipped}")


if __name__ == '__main__':
    main()