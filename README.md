# Distraction-Aware, Uncertainty-Calibrated Pedestrian Intention Prediction

### Pose Graph Transformers with Evidential Deep Learning

> Predict whether a pedestrian will cross the road **up to 4 seconds in advance** using skeleton pose, head orientation, and calibrated uncertainty — all from dashcam video.

---

## Highlights

- **0.782 F1@2s on JAAD** — outperforms PCPA (0.73) using skeleton features alone, without ego-vehicle speed or traffic light state
- **Calibrated uncertainty** via Evidential Deep Learning (ECE: 0.06) with temporal monotonicity constraint
- **Distraction-gated fusion** — head pose dynamically suppresses unreliable pose signals
- **Task-specific semantic edges** — biomechanically motivated graph topology for crossing prediction
- Evaluated on **JAAD + PIE** with full ablation study

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: 16 frames (1.33s)                     │
├────────────────────────────┬────────────────────────────────────┤
│    Stream A: Skeleton      │      Stream B: Head Pose           │
│    [16 × 17 × 4]          │      [16 × 3]                      │
│         │                  │           │                        │
│    Input Proj (4→64)       │      GRU (3→64)                    │
│    + Frame/Joint Embed     │           │                        │
│         │                  │      Distraction Context [64]      │
│    GATv2 Layer 1           │           │                        │
│    + Residual + LN + FFN   ├───────────┘                        │
│         │                  │                                    │
│    GATv2 Layer 2           │                                    │
│    + Residual + LN + FFN   │                                    │
│         │                  │                                    │
│    GATv2 Layer 3           │                                    │
│    + Residual + LN + FFN   │                                    │
│         │                  │                                    │
│    Mean Pool → [256]       │                                    │
│         │                  │                                    │
│    ┌────▼──────────────────▼────┐                               │
│    │  Distraction-Gated Fusion  │  gate = σ(W · distraction)    │
│    │  fused = gate ⊙ pose      │  + 0.3 · pose residual        │
│    │         + LayerNorm        │                               │
│    └────────────┬───────────────┘                               │
│                 │ [256]                                         │
│    ┌────────────┼────────────┬──────────────┐                  │
│    ▼            ▼            ▼              ▼                   │
│ Crossing    Distraction   EDL Uncertainty                       │
│ Head (4)    Head (2)      Head (α,β → p,u)                     │
│ P(cross     P(attentive/  Calibrated prob                      │
│ @0.5,1,     distracted)   + vacuity per                        │
│  2,4s)                    horizon                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Results

### Crossing Prediction (F1 @ threshold = 0.5)

| Method | JAAD F1@2s | PIE F1@2s | Uncertainty | Distraction |
|--------|:----------:|:---------:|:-----------:|:-----------:|
| SFRNN (Rasouli 2019) | — | ~0.62 | ✗ | ✗ |
| PedGraph (Cadena 2022) | ~0.70 | — | ✗ | ✗ |
| PCPA (Kotseruba 2021) | ~0.73 | ~0.73 | ✗ | ✗ |
| **Ours (Full Model)** | **0.782** | **0.709** | **✓** | **✓** |

### Ablation Study (Test Set)

| Stage | Configuration | JAAD F1@2s | PIE F1@2s | Avg |
|-------|--------------|:----------:|:---------:|:---:|
| 1 | Crossing loss only | 0.722 | 0.748 | 0.735 |
| 2 | + Distraction (w=0.05) | 0.789 | 0.709 | 0.749 |
| 3 | + EDL (w=0.05) | 0.798 | 0.679 | 0.739 |
| 4 | + Monotonicity (w=0.05) | 0.782 | 0.709 | 0.745 |

### Uncertainty Calibration

| Metric | Stage 1 (no EDL) | Stage 4 (full) |
|--------|:----------------:|:--------------:|
| ECE @2s | 0.131 | **0.061** |
| Monotonicity violations | 38.3% | **15.1%** |
| Uncertainty ordering | ✗ | 0.24 → 0.25 → 0.25 → 0.26 ✓ |

---

## Five Research Contributions

1. **Task-Specific Semantic Edges** — Foot-foot, wrist-wrist, and head-foot edges motivated by gait initiation biomechanics. First paper to design skeleton graph topology specifically for crossing prediction.

2. **Evidential Deep Learning + Temporal Monotonicity** — First application of EDL to pedestrian intention prediction. Novel monotonicity constraint ensures uncertainty increases at longer horizons.

3. **Distraction-Gated Pose Fusion** — Head pose generates an element-wise gate over the pose embedding, dynamically suppressing unreliable motion signals for distracted pedestrians.

4. **Multi-Task Joint Training** — Crossing prediction, distraction classification, and uncertainty estimation trained simultaneously on a shared backbone.

5. **Two-Dataset Evaluation** — Consistent evaluation on both JAAD and PIE, demonstrating generalization across datasets.

---

## Setup

### Prerequisites

- Python 3.10+
- macOS (Apple Silicon) or Linux with CUDA
- ~4GB disk space for preprocessed data

### Installation

```bash
git clone https://github.com/vasisht2004/pedestrian-intention-prediction.git
cd pedestrian-intention-prediction

conda create -n pedpred python=3.10 -y
conda activate pedpred
pip install -r requirements.txt
```

### Data

**Option A — Download preprocessed data (recommended)**

Download `npy_data/` from Google Drive:
[Download Link](https://drive.google.com/drive/folders/1HJw6lwJix4KwbulPpES4bWazAde8puAe?usp=drive_link)

Place it so the project has `npy_data/JAAD/` and `npy_data/PIE/` directories.

You also need the annotation XMLs (labels are resolved at runtime, not stored in .npy files):
- [JAAD annotations](https://github.com/ykotseruba/JAAD) — clone the repo
- [PIE annotations](https://github.com/aras62/PIE) — clone the repo

Update paths in `train.py`:
```python
JAAD_ROOT      = '/path/to/JAAD'
PIE_ANNOT_ROOT = '/path/to/PIE/annotations'
NPY_ROOT       = '/path/to/npy_data'
```

**Option B — Preprocess from raw videos**

```bash
python preprocess_mediapipe.py       # JAAD
python preprocess_mediapipe_pie.py   # PIE
```

---

## Usage

### Training (4-stage sequential)

```bash
# Stage 1: Crossing loss only (~20 min)
python train.py

# Stages 2-4: Adjust loss weights in losses.py, load previous checkpoint
# See train.py for checkpoint loading instructions
```

### Evaluation

```bash
python evaluate.py
```

Runs all stage checkpoints on the test set and produces F1, ECE, monotonicity analysis, and ablation summary.

---

## Project Structure

```
pedestrian-intention-prediction/
├── model.py                      # Full model architecture
│                                 #   GraphTransformerEncoder (3× GATv2)
│                                 #   GRUDistractionEncoder
│                                 #   DistractionGatedFusion
│                                 #   CrossingHead, DistractionHead, EDLHead
├── train.py                      # Training loop with staged loss activation
├── evaluate.py                   # Test set evaluation + ablation tables
├── losses.py                     # BCEWithLogits, Weighted CE, EDL, Monotonicity
├── dataset.py                    # PyTorch Dataset with quality filtering
├── graph_construction.py         # Skeleton graph: anatomical + temporal + semantic edges
├── jaad_parser.py                # JAAD XML annotation parser
├── pie_parser.py                 # PIE XML annotation parser
├── preprocess_mediapipe.py       # MediaPipe pose extraction (JAAD)
├── preprocess_mediapipe_pie.py   # MediaPipe pose extraction (PIE)
├── requirements.txt              # Python dependencies
└── checkpoints/                  # Saved model weights (not tracked)
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pose estimation | MediaPipe Pose (frozen) | Pretrained, no fine-tuning needed |
| Head pose | Face Mesh + solvePnP | Distance-invariant geometric angles, not learned |
| Graph network | GATv2 (not ST-GCN) | Dynamic attention over designed topology |
| Uncertainty | EDL (not MC Dropout) | Single forward pass, no ensemble needed |
| Fusion | Element-wise gate (not concat) | Per-dimension suppression of unreliable signals |
| Loss (Stage 1) | BCEWithLogitsLoss | Stronger gradients than Focal Loss on raw logits |
| Class imbalance | pos_weight=2.0 | Handles ~70% non-crossing majority class |

---

## Hardware

Developed and trained on Apple MacBook Air M4 (16GB unified memory) using PyTorch MPS backend. Full training pipeline (4 stages) completes in ~4 hours. Compatible with CUDA for GPU training.

---

## Known Limitations

- **Head pose data quality**: MediaPipe Face Mesh fails on 94% of distant pedestrians (>15m), limiting the distraction gate's effectiveness. The gating architecture is sound but requires more robust face detection for full benefit.
- **Skeleton signal**: Pre-crossing pose differences are subtle (velocity gap ~0.003 between crossers/non-crossers). Position features carry most discriminative signal.
- **Dataset size**: ~18K valid training samples after quality filtering. Larger datasets would benefit the graph attention layers.

---

## Citation

```bibtex
@inproceedings{vasisht2026pedestrian,
  title={Distraction-Aware, Uncertainty-Calibrated Pedestrian Intention Prediction 
         with Pose Graph Transformers},
  author={Vasisht, Payas},
  year={2026}
}
```

---

## Acknowledgments

- [JAAD Dataset](https://github.com/ykotseruba/JAAD) — Rasouli et al.
- [PIE Dataset](https://github.com/aras62/PIE) — Rasouli et al.
- [MediaPipe](https://mediapipe.dev/) — Google
- [PyTorch Geometric](https://pyg.org/) — Fey & Lenssen

---

## License

MIT