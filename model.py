"""
model.py

Full pedestrian intention prediction model.

Architecture:
    Stream A: skeleton graph [16,17,4] → Graph Transformer → [256] pose embedding
    Stream B: head pose [16,3]         → GRU encoder       → [64]  distraction context
    Fusion:   Distraction-Gated Fusion (Contribution 3)    → [256] fused embedding
    Heads:    Crossing (4) + Distraction (2) + EDL (2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from graph_construction import EDGE_INDEX, EDGE_TYPE, NUM_NODES, NUM_JOINTS, NUM_FRAMES


class GraphTransformerEncoder(nn.Module):
    """
    3-layer Graph Attention Network (GATv2) that processes the skeleton graph.
    """
    def __init__(self, in_dim=4, hidden_dim=64, out_dim=256, num_heads=4):
        super().__init__()

        self.input_proj  = nn.Linear(in_dim, hidden_dim)
        self.frame_embed = nn.Embedding(NUM_FRAMES, hidden_dim)
        self.joint_embed = nn.Embedding(NUM_JOINTS, hidden_dim)

        self.gat1 = GATv2Conv(hidden_dim,      hidden_dim // num_heads,
                               heads=num_heads, concat=True,  dropout=0.1)
        self.gat2 = GATv2Conv(hidden_dim,      hidden_dim // num_heads,
                               heads=num_heads, concat=True,  dropout=0.1)
        self.gat3 = GATv2Conv(hidden_dim,      hidden_dim // num_heads,
                               heads=num_heads, concat=False, dropout=0.1)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(),
                                   nn.Linear(hidden_dim*2, hidden_dim))
        self.ffn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(),
                                   nn.Linear(hidden_dim*2, hidden_dim))

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim // num_heads, out_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        batch_size = x.shape[0] // NUM_NODES

        x = self.input_proj(x)

        frame_ids = torch.arange(NUM_FRAMES, device=x.device).repeat_interleave(NUM_JOINTS).repeat(batch_size)
        joint_ids = torch.arange(NUM_JOINTS, device=x.device).repeat(NUM_FRAMES).repeat(batch_size)
        x = x + self.frame_embed(frame_ids) + self.joint_embed(joint_ids)

        h  = self.gat1(x, edge_index)
        h  = self.norm1(h + x)
        h  = h + self.ffn1(h)

        h2 = self.gat2(h, edge_index)
        h2 = self.norm2(h2 + h)
        h2 = h2 + self.ffn2(h2)

        h3 = self.gat3(h2, edge_index)
        h3 = F.relu(h3)

        h3     = h3.view(batch_size, NUM_NODES, -1)
        pooled = h3.mean(dim=1)

        out = self.output_proj(pooled)
        return out


class GRUDistractionEncoder(nn.Module):
    """
    Single-layer GRU that reads the 16-frame head pose sequence.
    """
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, headpose):
        _, hidden = self.gru(headpose)
        return hidden.squeeze(0)


class DistractionGatedFusion(nn.Module):
    """
    Contribution 3: Distraction-Gated Pose Fusion
    """
    def __init__(self, distraction_dim=64, pose_dim=256):
        super().__init__()
        self.gate_proj = nn.Linear(distraction_dim, pose_dim)
        self.norm      = nn.LayerNorm(pose_dim)

    def forward(self, pose_embedding, distraction_context):
        gate  = torch.sigmoid(self.gate_proj(distraction_context))
        fused = gate * pose_embedding + 0.3 * pose_embedding
        fused = self.norm(fused)
        return fused


class CrossingHead(nn.Module):
    """
    Predicts crossing probability at 4 time horizons.
    """
    def __init__(self, in_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DistractionHead(nn.Module):
    """
    Classifies pedestrian as attentive or distracted.
    """
    def __init__(self, in_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


class EDLUncertaintyHead(nn.Module):
    """
    Contribution 2: Evidential Deep Learning uncertainty head.
    """
    def __init__(self, in_dim=256, num_horizons=4):
        super().__init__()
        self.net          = nn.Linear(in_dim, num_horizons * 2)
        self.num_horizons = num_horizons

    def forward(self, x):
        raw  = self.net(x)
        raw  = raw.view(-1, self.num_horizons, 2)

        alpha       = F.softplus(raw[:, :, 0]) + 1
        beta        = F.softplus(raw[:, :, 1]) + 1
        prob        = alpha / (alpha + beta)
        uncertainty = 2.0  / (alpha + beta)

        return alpha, beta, prob, uncertainty


class PedestrianIntentModel(nn.Module):
    """
    Full model combining all components.
    """
    def __init__(self):
        super().__init__()

        self.graph_encoder    = GraphTransformerEncoder(in_dim=4, hidden_dim=64,
                                                        out_dim=256, num_heads=4)
        self.gru_encoder      = GRUDistractionEncoder(input_dim=3, hidden_dim=64)
        self.fusion           = DistractionGatedFusion(distraction_dim=64, pose_dim=256)
        self.crossing_head    = CrossingHead(in_dim=256)
        self.distraction_head = DistractionHead(in_dim=256)
        self.edl_head         = EDLUncertaintyHead(in_dim=256, num_horizons=4)

    def forward(self, skeleton, headpose, edge_index):
        """
        skeleton:   [batch, 16, 17, 4]
        headpose:   [batch, 16, 3]
        edge_index: [2, num_edges]
        """
        batch_size = skeleton.shape[0]

        # Stream A
        x                  = skeleton.view(batch_size * 272, 4)
        batch_edge_index   = self._batch_edge_index(edge_index, batch_size, skeleton.device)
        pose_embedding     = self.graph_encoder(x, batch_edge_index)

        # Stream B
        distraction_context = self.gru_encoder(headpose)

        # Fusion
        fused = self.fusion(pose_embedding, distraction_context)

        # Heads
        crossing_probs     = self.crossing_head(fused)
        distraction_logits = self.distraction_head(fused)
        alpha, beta, edl_probs, uncertainty = self.edl_head(fused)

        return {
            'crossing_probs':     crossing_probs,
            'distraction_logits': distraction_logits,
            'alpha':              alpha,
            'beta':               beta,
            'edl_probs':          edl_probs,
            'uncertainty':        uncertainty,
        }

    def _batch_edge_index(self, edge_index, batch_size, device):
        edge_index = edge_index.to(device)
        if batch_size == 1:
            return edge_index
        edge_indices = []
        for i in range(batch_size):
            edge_indices.append(edge_index + i * 272)
        return torch.cat(edge_indices, dim=1)


if __name__ == '__main__':
    from graph_construction import EDGE_INDEX

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model      = PedestrianIntentModel().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    batch_size = 4
    skeleton   = torch.rand(batch_size, 16, 17, 4).to(device)
    headpose   = torch.rand(batch_size, 16, 3).to(device)
    edge_index = EDGE_INDEX.to(device)

    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(skeleton, headpose, edge_index)

    print(f"crossing_probs shape:     {outputs['crossing_probs'].shape}")
    print(f"distraction_logits shape: {outputs['distraction_logits'].shape}")
    print(f"edl_probs shape:          {outputs['edl_probs'].shape}")
    print(f"uncertainty shape:        {outputs['uncertainty'].shape}")
    print(f"\nSample crossing probs: {outputs['crossing_probs'][0]}")
    print(f"Sample uncertainty:    {outputs['uncertainty'][0]}")
    print(f"Sample EDL probs:      {outputs['edl_probs'][0]}")