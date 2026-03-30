import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import sys

sys.path.append('/Users/payas/pedestrian_project')
from jaad_parser import parse_jaad_dataset

# ── paths ──────────────────────────────────────────────────────────────────────
JAAD_CLIPS_DIR = '/Users/payas/JAAD/JAAD_clips'
OUTPUT_DIR     = '/Users/payas/pedestrian_project/npy_data/JAAD'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_pose      = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# 6 canonical face points for solvePnP (nose, chin, left eye, right eye, left mouth, right mouth)
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0   ],   # nose tip
    [0.0,   -63.6,  -12.5 ],   # chin
    [-43.3,  32.7,  -26.0 ],   # left eye corner
    [43.3,   32.7,  -26.0 ],   # right eye corner
    [-28.9, -28.9,  -24.1 ],   # left mouth corner
    [28.9,  -28.9,  -24.1 ],   # right mouth corner
], dtype=np.float64)

FACE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]  # corresponding MediaPipe ids

def get_head_pose(landmarks_2d, frame_w, frame_h):
    """Run solvePnP to get yaw, pitch, roll from 6 face landmarks."""
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


def process_sample(sample, pose_model, face_model):
    video_path = os.path.join(JAAD_CLIPS_DIR, sample['video_id'] + '.mp4')
    if not os.path.exists(video_path):
        return None, None

    cap = cv2.VideoCapture(video_path)
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

        bw = xbr - xtl
        bh = ybr - ytl
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # ── Pose ──────────────────────────────────────────────────────────────
        pose_result = pose_model.process(crop_rgb)
        joints = np.zeros((17, 4), dtype=np.float32)

        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks.landmark
            # MediaPipe 33 landmarks → we keep indices 0,11-16,23-28 = 17 joints
            keep = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            for j, idx in enumerate(keep):
                lm = lms[idx]
                if lm.visibility > 0.5:
                    joints[j, 0] = lm.x   # already normalised 0-1 by MediaPipe
                    joints[j, 1] = lm.y
                else:
                    # forward fill
                    if prev_joints is not None:
                        joints[j, 0] = prev_joints[j, 0]
                        joints[j, 1] = prev_joints[j, 1]

        # velocities
        if prev_joints is not None:
            joints[:, 2] = joints[:, 0] - prev_joints[:, 0]  # vx
            joints[:, 3] = joints[:, 1] - prev_joints[:, 1]  # vy

        skeleton_seq[t] = joints
        prev_joints     = joints.copy()

        # ── Head pose ─────────────────────────────────────────────────────────
        face_result = face_model.process(crop_rgb)
        if face_result.multi_face_landmarks:
            fl = face_result.multi_face_landmarks[0].landmark
            pts_2d = np.array([
                [fl[i].x * bw, fl[i].y * bh] for i in FACE_LANDMARK_IDS
            ], dtype=np.float64)
            headpose_seq[t] = get_head_pose(pts_2d, bw, bh)
        else:
            if t > 0:
                headpose_seq[t] = headpose_seq[t-1]  # forward fill

    cap.release()
    return skeleton_seq, headpose_seq


def main():
    print("Loading JAAD samples...")
    samples = parse_jaad_dataset('/Users/payas/JAAD')
    print(f"Total samples: {len(samples)}")

    skipped  = 0
    saved    = 0

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose_model, \
         mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_model:

        for sample in tqdm(samples, desc="Preprocessing"):
            vid  = sample['video_id']
            pid  = sample['ped_id'].replace('/', '-')
            sf   = sample['start_frame']
            stem = f"{vid}_{pid}_{sf}"

            skel_path = os.path.join(OUTPUT_DIR, stem + '_skeleton.npy')
            head_path = os.path.join(OUTPUT_DIR, stem + '_headpose.npy')

            # skip if already done
            if os.path.exists(skel_path) and os.path.exists(head_path):
                saved += 1
                continue

            skeleton, headpose = process_sample(sample, pose_model, face_model)

            if skeleton is None:
                skipped += 1
                continue

            np.save(skel_path, skeleton)
            np.save(head_path, headpose)
            saved += 1

    print(f"\nDone. Saved: {saved}  Skipped: {skipped}")


if __name__ == '__main__':
    main()