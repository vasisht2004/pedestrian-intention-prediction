# Pedestrian Intention Prediction

Predicts whether a pedestrian will cross a street at four time horizons (0.5s, 1s, 2s, 4s) using skeleton pose and head pose data from the JAAD and PIE datasets.

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/pedestrian-intention-prediction.git
cd pedestrian-intention-prediction

# 2. Create and activate conda environment
conda create -n pedpred python=3.10
conda activate pedpred

# 3. Install dependencies
pip install -r requirements.txt

# 4. Get the preprocessed data
# Download npy_data/ from Google Drive: <PLACEHOLDER_LINK>
# Extract it so that npy_data/JAAD/ and npy_data/PIE/ exist in the project root.

# 5. Train
python train.py
```

## Dataset Setup

You have two options for the preprocessed skeleton + head pose data:

**Option A — Download preprocessed data (recommended)**
Download `npy_data/` from the shared Google Drive folder: `<PLACEHOLDER_LINK>`

**Option B — Preprocess from raw videos**
1. Download the [JAAD dataset](https://github.com/ykotseruba/JAAD) and point `JAAD_ROOT` in `train.py` to it.
2. Download the [PIE dataset](https://github.com/aras62/PIE) and point `PIE_ANNOT_ROOT` in `train.py` to it.
3. Run the preprocessing scripts:
   ```bash
   python preprocess_mediapipe.py       # JAAD
   python preprocess_mediapipe_pie.py   # PIE
   ```

## File Reference

| File | Description |
|------|-------------|
| `train.py` | Main training loop: data loading, optimizer, loss, early stopping, checkpointing |
| `model.py` | Full model architecture: Skeleton Graph Transformer + Head Pose GRU + Distraction-Gated Fusion + task heads |
| `losses.py` | Four-component loss: Focal, Weighted CE, EDL uncertainty, Temporal Monotonicity |
| `dataset.py` | PyTorch Dataset combining JAAD + PIE; handles splits, augmentation, and output formatting |
| `graph_construction.py` | Builds the skeleton graph with anatomical, temporal, and semantic edge types |
| `jaad_parser.py` | Parses JAAD XML annotations into 16-frame observation windows with crossing labels |
| `pie_parser.py` | Parses PIE XML annotations into observation windows, with frame-rate normalization |
| `preprocess_mediapipe.py` | Extracts MediaPipe skeleton and head pose from JAAD video clips; writes `.npy` files |
| `preprocess_mediapipe_pie.py` | Same as above for PIE videos, with multiprocessing support |
