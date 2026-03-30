# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pedestrian intention prediction system that classifies whether pedestrians will cross a street at four time horizons (0.5s, 1s, 2s, 4s). Uses skeleton pose + head pose data extracted from JAAD and PIE datasets via MediaPipe.

## Running the Code

**Train the model:**
```bash
python train.py
```

**Smoke-test individual modules (each has a `if __name__ == "__main__"` block):**
```bash
python model.py           # forward pass with dummy tensors
python dataset.py         # dataset loading check
python losses.py          # all loss function checks
python graph_construction.py  # graph builder check
```

**Preprocessing (run once before training):**
```bash
python preprocess_mediapipe.py        # JAAD: extracts skeleton + head pose → npy_data/JAAD/
python preprocess_mediapipe_pie.py    # PIE:  extracts skeleton + head pose → npy_data/PIE/
```

There is no requirements.txt. Core dependencies: `torch`, `torch_geometric`, `numpy`, `cv2`, `mediapipe`, `sklearn`, `tqdm`.

Hardware: training auto-selects MPS (Apple Silicon) then CPU (`DEVICE` in `train.py`).

## Architecture

The model (`model.py`) has two parallel streams that fuse before task heads:

**Stream A — Skeleton Graph Transformer**
- Input: `[B, 16, 17, 4]` — 16 frames × 17 MediaPipe joints × (x_norm, y_norm, vx, vy)
- Graph nodes: 272 (16 frames × 17 joints); three edge types built in `graph_construction.py`:
  1. *Anatomical* — skeletal bones within a frame
  2. *Temporal* — same joint across consecutive frames
  3. *Semantic* — biomechanically motivated (foot-foot, wrist-wrist, head-foot)
- 3-layer GATv2; mean-pool → `[B, 256]` pose embedding

**Stream B — Head Pose GRU**
- Input: `[B, 16, 3]` — yaw/pitch/roll per frame
- Single-layer GRU → `[B, 64]` distraction context

**Distraction-Gated Fusion** merges both streams → `[B, 256]`

**Task heads (multi-task):**
| Head | Output | Purpose |
|------|--------|---------|
| Crossing | `[B, 4]` | Crossing probability at each horizon |
| Distraction | `[B, 2]` | Attentive / distracted logits |
| EDL Uncertainty | `[B, 4]` each for α, β, prob, uncertainty | Evidential uncertainty per horizon |

## Loss Function (`losses.py`)

Four-component combined loss:
1. **Focal Loss** (w=1.0) — crossing classification, γ=2.0
2. **Weighted Cross-Entropy** (w=0.5) — distraction head; class weights `[1.0, 3.0]`
3. **EDL Loss** (w=0.5) — evidential fit term + KL regularization, annealed over 10 epochs
4. **Temporal Monotonicity Loss** (w=0.1) — enforces uncertainty increases with horizon

## Data Pipeline

**Preprocessing** writes one `.npy` file per frame per pedestrian track into `npy_data/JAAD/` or `npy_data/PIE/`. Each file stores either skeleton (17×4) or headpose (3,).

**Parsing** (`jaad_parser.py`, `pie_parser.py`) reads XML annotations and assembles 16-frame sliding windows with crossing labels at the four horizons.

**Dataset class** (`dataset.py`) merges JAAD + PIE, splits by `video_id` (15% val, 15% test — no leakage), applies Gaussian noise + horizontal flip augmentation during training, and returns `(skeleton, headpose, crossing_labels, distraction_label, dataset_flag, has_distraction)`.

PIE is sampled every 2–3 frames to match JAAD's ~10 fps effective rate.

## Key Hardcoded Paths (in `train.py`)

```python
JAAD_ROOT      = '/Users/payas/JAAD'
PIE_ANNOT_ROOT = '/Users/payas/PIE_annotations/annotations/annotations'
NPY_ROOT       = '/Users/payas/pedestrian_project/npy_data'
CHECKPOINT_DIR = '/Users/payas/pedestrian_project/checkpoints'
```

Best checkpoint saved to `checkpoints/best_model.pt` based on F1@2s on validation set.

## Training Hyperparameters

```python
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR         = 1e-4   # cosine-annealed to 1e-6
WEIGHT_DECAY = 1e-4
PATIENCE   = 10     # early stopping on F1@2s
HORIZON_IDX = 2     # 2s horizon used for model selection
```
Optimizer: AdamW. Gradient clipping: `max_norm=1.0`.
