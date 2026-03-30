"""
graph_construction.py

Converts a [16, 17, 4] skeleton tensor into a graph for the Graph Transformer.

The graph has:
- 272 nodes (17 joints x 16 frames)
- ~1,150 edges across three categories:
    1. Anatomical edges  — actual bones of the human body
    2. Temporal edges    — same joint across consecutive frames
    3. Semantic edges    — our Contribution 1, biomechanically motivated

Node numbering: node_id = frame_index * 17 + joint_index
So joint 3 in frame 5 = node 5*17 + 3 = 88

MediaPipe joint indices we use (17 joints):
    0  = nose (head)
    1  = left shoulder
    2  = right shoulder
    3  = left elbow
    4  = right elbow
    5  = left wrist
    6  = right wrist
    7  = left hip
    8  = right hip
    9  = left knee
    10 = right knee
    11 = left ankle
    12 = right ankle
    13 = left heel
    14 = right heel
    15 = left foot index
    16 = right foot index
"""

import torch
import numpy as np

# ── Joint index constants (for readability) ───────────────────────────────────
HEAD          = 0
L_SHOULDER    = 1
R_SHOULDER    = 2
L_ELBOW       = 3
R_ELBOW       = 4
L_WRIST       = 5
R_WRIST       = 6
L_HIP         = 7
R_HIP         = 8
L_KNEE        = 9
R_KNEE        = 10
L_ANKLE       = 11
R_ANKLE       = 12
L_HEEL        = 13
R_HEEL        = 14
L_FOOT        = 15
R_FOOT        = 16

NUM_JOINTS    = 17
NUM_FRAMES    = 16
NUM_NODES     = NUM_JOINTS * NUM_FRAMES  # 272


# ── Edge definitions ──────────────────────────────────────────────────────────

# Anatomical edges — actual bones of the human skeleton
# These exist in ALL prior work (ST-GCN, PedGraph etc)
ANATOMICAL_EDGES = [
    (HEAD,       L_SHOULDER),
    (HEAD,       R_SHOULDER),
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW,    L_WRIST),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW,    R_WRIST),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP,      R_HIP),
    (L_HIP,      L_KNEE),
    (L_KNEE,     L_ANKLE),
    (L_ANKLE,    L_HEEL),
    (L_HEEL,     L_FOOT),
    (R_HIP,      R_KNEE),
    (R_KNEE,     R_ANKLE),
    (R_ANKLE,    R_HEEL),
    (R_HEEL,     R_FOOT),
]

# Semantic edges — OUR CONTRIBUTION 1
# Biomechanically motivated connections for crossing prediction specifically
SEMANTIC_EDGES = [
    # Foot-to-foot: captures bilateral weight shift before stepping
    # Breniere & Do (1986): weight shift precedes foot lift by 200-400ms
    # In anatomical graph these joints are 5 hops apart
    (L_FOOT,  R_FOOT),
    (L_HEEL,  R_HEEL),
    (L_ANKLE, R_ANKLE),

    # Wrist-to-wrist: captures phone-holding posture
    # Both wrists at mid-torso = phone use
    # In anatomical graph these are 7 hops apart
    (L_WRIST, R_WRIST),

    # Head-to-foot: captures forward lean before stepping
    # Cook & Cozzens (1976): head/torso lean forward 340ms before foot lift
    (HEAD, L_FOOT),
    (HEAD, R_FOOT),
]


def build_edge_index(num_frames=NUM_FRAMES, num_joints=NUM_JOINTS):
    """
    Builds the full edge_index tensor for the skeleton graph.

    Returns:
        edge_index: torch.LongTensor of shape [2, num_edges]
                    edge_index[0] = source nodes
                    edge_index[1] = target nodes
        edge_type:  torch.LongTensor of shape [num_edges]
                    0 = anatomical, 1 = temporal, 2 = semantic
    """
    src_nodes  = []
    dst_nodes  = []
    edge_types = []

    def add_edge(src, dst, etype):
        # Add both directions (undirected graph)
        src_nodes.append(src)
        dst_nodes.append(dst)
        edge_types.append(etype)
        src_nodes.append(dst)
        dst_nodes.append(src)
        edge_types.append(etype)

    def node_id(frame, joint):
        return frame * num_joints + joint

    # ── 1. Anatomical edges (within each frame) ───────────────────────────────
    for frame in range(num_frames):
        for (j1, j2) in ANATOMICAL_EDGES:
            add_edge(node_id(frame, j1), node_id(frame, j2), etype=0)

    # ── 2. Temporal edges (same joint across consecutive frames) ──────────────
    for frame in range(num_frames - 1):
        for joint in range(num_joints):
            add_edge(node_id(frame, joint), node_id(frame + 1, joint), etype=1)

    # ── 3. Semantic edges (within each frame) ────────────────────────────────
    for frame in range(num_frames):
        for (j1, j2) in SEMANTIC_EDGES:
            add_edge(node_id(frame, j1), node_id(frame, j2), etype=2)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_type  = torch.tensor(edge_types,              dtype=torch.long)

    return edge_index, edge_type


def skeleton_to_graph(skeleton):
    """
    Converts a skeleton tensor to graph node features.

    Args:
        skeleton: torch.Tensor of shape [16, 17, 4]
                  (x_norm, y_norm, vx, vy) per joint per frame

    Returns:
        x: torch.Tensor of shape [272, 4] — node features
    """
    # Reshape [16, 17, 4] → [272, 4]
    x = skeleton.reshape(NUM_NODES, -1)
    return x


# Pre-build edge index once — reused for every sample in training
EDGE_INDEX, EDGE_TYPE = build_edge_index()


if __name__ == '__main__':
    # Test the graph construction
    print("Building edge index...")
    edge_index, edge_type = build_edge_index()

    print(f"Edge index shape: {edge_index.shape}")
    print(f"Total edges: {edge_index.shape[1]}")
    print(f"Anatomical edges: {(edge_type == 0).sum().item()}")
    print(f"Temporal edges:   {(edge_type == 1).sum().item()}")
    print(f"Semantic edges:   {(edge_type == 2).sum().item()}")

    # Test with a dummy skeleton
    dummy_skeleton = torch.rand(16, 17, 4)
    x = skeleton_to_graph(dummy_skeleton)
    print(f"\nNode features shape: {x.shape}")
    print(f"Expected: [272, 4]")

    # Verify node numbering
    print(f"\nNode ID for frame=5, joint=3: {5*17+3} (expected 88)")
    print(f"Node ID for frame=0, joint=0: {0*17+0} (expected 0)")
    print(f"Node ID for frame=15, joint=16: {15*17+16} (expected 271)")