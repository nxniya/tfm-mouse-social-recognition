"""
src/models/graph.py
====================
A GNN plus BiLSTM model for behaviour classification over the mouse's
skeletal graph.

Provides:
- ``SkeletonGraph``      — the articulated CalMS21 (7 kp) / MABe22 (12 kp) graph,
                           with ``edge_index`` held as a PyTorch buffer
- ``BehaviorSkeletonGNN``— two GATv2Conv layers per frame, then a temporal
                           BiLSTM and a classifier; the M4 architecture

Data flow::

    kp_seq : (N, T, n_nodes, n_feat)   per-mouse keypoint sequence
        ↓
    [reshape] (N·T, n_nodes, n_feat)
        ↓
    [GATv2Conv × n_gat_layers]  → node embeddings (N·T, n_nodes, d_gnn)
        ↓
    [global mean pool]          → graph embedding  (N·T, d_gnn)
        ↓
    [reshape] (N, T, d_gnn)     × 2 mice  → concat (N, T, 2·d_gnn)
        ↓
    [BiLSTM × lstm_layers + TemporalAttention]
        ↓
    [LayerNorm + Dropout → Linear]  → logits (N, n_classes)

``torch_geometric`` (for GATv2Conv) is an optional dependency. When it is not
installed, GraphAttentionFallback provides a pure multi-head attention
implementation that needs nothing beyond PyTorch.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional import of torch_geometric
_TG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None

if _TG_AVAILABLE:
    try:
        from torch_geometric.nn import GATv2Conv as _GATv2Conv
    except Exception:
        _TG_AVAILABLE = False

# TemporalAttention is reused from rnn.py
from src.models.rnn import TemporalAttention


# ---------------------------------------------------------------------------
# The articulated mouse graph
# ---------------------------------------------------------------------------

# CalMS21 keypoints (7 kp); the index order is the canonical one, shared with
# MouseSkeleton, so the two modules can be used together
CALMS21_KEYPOINTS: List[str] = [
    "nose",       # 0
    "neck",       # 1
    "ear_left",   # 2
    "ear_right",  # 3
    "hip_left",   # 4
    "hip_right",  # 5
    "tail_base",  # 6
]

# Edges of the CalMS21 skeleton
_CALMS21_EDGES: List[Tuple[int, int]] = [
    (0, 1),  # nose → neck
    (1, 2),  # neck → ear_left
    (1, 3),  # neck → ear_right
    (1, 4),  # neck → hip_left
    (1, 5),  # neck → hip_right
    (4, 6),  # hip_left → tail_base
    (5, 6),  # hip_right → tail_base
]

# MABe22 keypoints (12 kp), a superset of CalMS21
MABE22_KEYPOINTS: List[str] = [
    "nose",           # 0
    "neck",           # 1
    "ear_left",       # 2
    "ear_right",      # 3
    "body_center",    # 4
    "hip_left",       # 5
    "hip_right",      # 6
    "forepaw_left",   # 7
    "forepaw_right",  # 8
    "hindpaw_left",   # 9
    "hindpaw_right",  # 10
    "tail_base",      # 11
]

_MABE22_EDGES: List[Tuple[int, int]] = [
    (0, 1),   # nose → neck
    (1, 2),   # neck → ear_left
    (1, 3),   # neck → ear_right
    (1, 4),   # neck → body_center
    (4, 5),   # body_center → hip_left
    (4, 6),   # body_center → hip_right
    (4, 7),   # body_center → forepaw_left
    (4, 8),   # body_center → forepaw_right
    (5, 9),   # hip_left → hindpaw_left
    (6, 10),  # hip_right → hindpaw_right
    (4, 11),  # body_center → tail_base
]


class SkeletonGraph:
    """The articulated graph, for either of the two keypoint sets.

    ``edge_index`` follows the PyTorch Geometric convention: a tensor of shape
    ``(2, num_edges)`` holding **undirected** edges, meaning both directions are
    present.

    Parameters
    ----------
    lab_type : str
        ``"calms21"`` (7 kp) or ``"mabe22"`` (12 kp).
    device : torch.device, optional
        Device on which to create the tensors. Defaults to ``cpu``.
    """

    def __init__(self, lab_type: str = "calms21", device: Optional[torch.device] = None) -> None:
        self.lab_type = lab_type.lower()
        self.device = device or torch.device("cpu")

        if self.lab_type == "calms21":
            self.keypoints = CALMS21_KEYPOINTS
            raw_edges = _CALMS21_EDGES
        elif self.lab_type in ("mabe22", "mabe22_keypoints"):
            self.keypoints = MABE22_KEYPOINTS
            raw_edges = _MABE22_EDGES
        else:
            raise ValueError(f"Unknown lab_type: {lab_type!r}. Use 'calms21' or 'mabe22'.")

        self.n_nodes: int = len(self.keypoints)
        self.kp_index: Dict[str, int] = {kp: i for i, kp in enumerate(self.keypoints)}

        # Undirected edges: include both src to dst and dst to src
        src = [e[0] for e in raw_edges] + [e[1] for e in raw_edges]
        dst = [e[1] for e in raw_edges] + [e[0] for e in raw_edges]
        self._edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)

    @property
    def edge_index(self) -> torch.Tensor:
        """Tensor of shape ``(2, 2 * num_edges)`` holding undirected edges."""
        return self._edge_index

    def to(self, device: torch.device) -> "SkeletonGraph":
        """Move ``edge_index`` to the given device."""
        self.device = device
        self._edge_index = self._edge_index.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"SkeletonGraph(lab_type={self.lab_type!r}, "
            f"n_nodes={self.n_nodes}, n_edges={self._edge_index.shape[1] // 2})"
        )


# ---------------------------------------------------------------------------
# Fallback: multi-head attention without torch_geometric
# ---------------------------------------------------------------------------

class _GraphAttentionFallback(nn.Module):
    """A lightweight GAT that needs no torch_geometric; it uses dense matrices.

    With small node sets (14 or fewer) and a fixed graph, a dense matrix
    multiplication is faster than a sparse implementation, so nothing is lost by
    dropping the dependency.

    The simplified GAT layer:
        e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
        alpha_{ij} = softmax_j(e_{ij})   over graph neighbours only
        h'_i   = sigma(sum_j alpha_{ij} * Wh_j), heads concatenated

    Parameters
    ----------
    in_channels : int
        Input width per node.
    out_channels : int
        Output width **per head**.
    n_heads : int
        Number of attention heads.
    concat : bool
        When True, concatenate the heads, giving out_channels * n_heads.
        When False, average them, giving out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.out_channels = out_channels
        self.concat = concat

        # A single linear projection shared by every head
        self.W = nn.Linear(in_channels, out_channels * n_heads, bias=False)
        # The attention vector a, scored through a LeakyReLU
        self.a = nn.Parameter(torch.empty(1, n_heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.a)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)
        self.act = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        adj_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (B, n_nodes, in_channels)
        adj_mask : (n_nodes, n_nodes)  boolean, True where an edge exists

        Returns
        -------
        (B, n_nodes, out_channels * n_heads) when concat, else out_channels
        """
        B, N, _ = x.shape
        Wh = self.W(x)                          # (B, N, H·d)
        Wh = Wh.view(B, N, self.n_heads, self.out_channels)  # (B, N, H, d)

        # Attention scores
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, N, N, H, d)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1, -1)  # (B, N, N, H, d)
        cat  = torch.cat([Wh_i, Wh_j], dim=-1)             # (B, N, N, H, 2d)

        # a shape: (1, 1, 1, H, 2d) — broadcast over B, N, N
        e = self.leaky((self.a.unsqueeze(0).unsqueeze(0) * cat).sum(-1))  # (B, N, N, H)

        # Mask: set non-existent edges to -inf before the softmax
        mask = adj_mask.unsqueeze(0).unsqueeze(-1)          # (1, N, N, 1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=2)                     # (B, N, N, H)
        alpha = self.dropout(alpha)

        # Aggregate the messages
        # (B, N, N, H) × (B, N, H, d) → (B, N, H, d)
        out = torch.einsum("bnih,bjhd->bihd", alpha, Wh)   # (B, N, H, d)

        if self.concat:
            out = out.reshape(B, N, self.n_heads * self.out_channels)
        else:
            out = out.mean(dim=2)                           # (B, N, d)

        return self.act(out)


# ---------------------------------------------------------------------------
# BehaviorSkeletonGNN — modelo M4
# ---------------------------------------------------------------------------

class BehaviorSkeletonGNN(nn.Module):
    """Per-frame GATv2 followed by a temporal BiLSTM, for behaviour classification.

    Combines the structural information in the skeletal graph, that is, which
    keypoints are connected, with the temporal dynamics of the window.

    Flow::

        kp_seq_m1 : (N, T, n_nodes, n_feat)  → mouse 1
        kp_seq_m2 : (N, T, n_nodes, n_feat)  → mouse 2
              ↓
        [flatten time]  (N·T, n_nodes, n_feat)  for each mouse
              ↓
        [GATv2Conv × n_gat_layers]  → (N·T, n_nodes, d_gnn)
              ↓
        [global mean pool over nodes]  → (N·T, d_gnn)
              ↓
        [reshape + concat mice]  → (N, T, 2·d_gnn)
              ↓
        [BiLSTM × lstm_layers + TemporalAttention]
              ↓
        [LayerNorm + Dropout → Linear]  → logits (N, n_classes)

    Parameters
    ----------
    n_feat : int
        Features per node per frame. Typically 5: [x, y, vx, vy, dz].
    n_nodes : int
        Nodes per mouse: 7 for CalMS21, 12 for MABe22.
    d_gnn : int
        GNN embedding width per node, per head.
    n_gat_layers : int
        Number of GATv2 layers.
    n_gat_heads : int
        Attention heads in each GATv2 layer.
    lstm_hidden : int
        LSTM units per direction.
    lstm_layers : int
        Stacked BiLSTM layers.
    n_classes : int
        Number of behaviour classes.
    dropout : float
        Global dropout.
    lab_type : str
        ``"calms21"`` or ``"mabe22"``, which selects the skeletal graph.
    """

    def __init__(
        self,
        n_feat: int = 5,
        n_nodes: int = 7,
        d_gnn: int = 64,
        n_gat_layers: int = 2,
        n_gat_heads: int = 4,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        n_classes: int = 8,
        dropout: float = 0.3,
        lab_type: str = "calms21",
    ) -> None:
        super().__init__()

        self.n_nodes = n_nodes
        self.d_gnn = d_gnn
        self.n_gat_heads = n_gat_heads
        self.lab_type = lab_type
        self.use_pyg = _TG_AVAILABLE

        # --- Skeletal graph --------------------------------------------------
        self.skeleton = SkeletonGraph(lab_type=lab_type)

        if not self.use_pyg:
            warnings.warn(
                "torch_geometric is not available; "
                "falling back to GraphAttentionFallback, a dense GAT.",
                stacklevel=2,
            )

        # --- Input projection ------------------------------------------------
        # Project n_feat up to d_gnn so that the first GAT layer has a sensible
        # width whatever n_feat happens to be
        self.input_proj = nn.Sequential(
            nn.Linear(n_feat, d_gnn),
            nn.LayerNorm(d_gnn),
            nn.GELU(),
        )

        # --- GATv2 layers, or the fallback ----------------------------------
        gat_in = d_gnn            # after the input projection
        gat_out_per_head = d_gnn  # per head
        gat_out_total = gat_out_per_head * n_gat_heads  # each layer concatenates

        self.gat_layers = nn.ModuleList()
        for i in range(n_gat_layers):
            in_ch = gat_in if i == 0 else gat_out_total
            if self.use_pyg:
                # GATv2Conv with concat=True gives out = heads * out_channels
                layer = _GATv2Conv(
                    in_channels=in_ch,
                    out_channels=gat_out_per_head,
                    heads=n_gat_heads,
                    concat=True,
                    dropout=dropout * 0.5,
                )
            else:
                layer = _GraphAttentionFallback(
                    in_channels=in_ch,
                    out_channels=gat_out_per_head,
                    n_heads=n_gat_heads,
                    concat=True,
                    dropout=dropout * 0.5,
                )
            self.gat_layers.append(layer)

        # Final projection of the graph embedding back down to d_gnn
        self.gnn_proj = nn.Sequential(
            nn.Linear(gat_out_total, d_gnn),
            nn.LayerNorm(d_gnn),
            nn.GELU(),
        )

        # --- Temporal layer: BiLSTM ------------------------------------------
        # Input is the concatenation of both mice's embeddings, so 2 * d_gnn
        lstm_in = 2 * d_gnn
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        d_out = lstm_hidden * 2  # bidireccional
        self.attention = TemporalAttention(d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_out, n_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_adj_mask(self, device: torch.device) -> torch.Tensor:
        """Build the boolean adjacency mask, shape (n_nodes, n_nodes)."""
        n = self.n_nodes
        adj = torch.zeros(n, n, dtype=torch.bool, device=device)
        ei = self.skeleton.edge_index.to(device)
        adj[ei[0], ei[1]] = True
        # Include self-connections; without them the GNN is unstable
        adj[torch.arange(n, device=device), torch.arange(n, device=device)] = True
        return adj

    def _gnn_encode(self, kp_seq: torch.Tensor) -> torch.Tensor:
        """Encode (N, T, n_nodes, n_feat) into (N, T, d_gnn).

        Parameters
        ----------
        kp_seq : Tensor (N, T, n_nodes, n_feat)

        Returns
        -------
        Tensor (N, T, d_gnn)
        """
        N, T, V, n_feat_in = kp_seq.shape
        device = kp_seq.device

        # Flatten time so every frame is processed in parallel, as one batch
        x = kp_seq.reshape(N * T, V, n_feat_in)           # (N·T, V, F)
        x = self.input_proj(x)                             # (N·T, V, d_gnn)

        if self.use_pyg:
            # torch_geometric expects (N_total, F) with a single graph's
            # edge_index. Since every graph in the batch is identical, build the
            # batched edge_index by offsetting one copy per graph
            ei_single = self.skeleton.edge_index.to(device)  # (2, E)
            batch_size = N * T
            offsets = torch.arange(batch_size, device=device).unsqueeze(1) * V  # (B, 1)
            ei_batch = (ei_single.unsqueeze(0) + offsets.unsqueeze(-1))         # (B, 2, E)
            ei_batch = ei_batch.permute(1, 0, 2).reshape(2, -1)                 # (2, B·E)

            x_flat = x.reshape(N * T * V, -1)             # (N·T·V, d_gnn)
            for gat in self.gat_layers:
                x_flat = F.elu(gat(x_flat, ei_batch))
            x = x_flat.reshape(N * T, V, -1)              # (N·T, V, gat_out_total)
        else:
            adj_mask = self._build_adj_mask(device)        # (V, V)
            for gat in self.gat_layers:
                x = gat(x, adj_mask)                       # (N·T, V, gat_out_total)

        # Global mean pooling over the nodes
        x = x.mean(dim=1)                                  # (N·T, gat_out_total)
        x = self.gnn_proj(x)                               # (N·T, d_gnn)
        return x.reshape(N, T, self.d_gnn)                 # (N, T, d_gnn)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def get_embedding(
        self,
        kp_seq_m1: torch.Tensor,
        kp_seq_m2: torch.Tensor,
    ) -> torch.Tensor:
        """The embedding taken before the classification head, for Coreset AL.

        Parameters
        ----------
        kp_seq_m1 : Tensor (N, T, n_nodes, n_feat)  — mouse 1
        kp_seq_m2 : Tensor (N, T, n_nodes, n_feat)  — mouse 2

        Returns
        -------
        embedding : Tensor (N, 2·lstm_hidden)
        """
        emb1 = self._gnn_encode(kp_seq_m1)                 # (N, T, d_gnn)
        emb2 = self._gnn_encode(kp_seq_m2)                 # (N, T, d_gnn)
        combined = torch.cat([emb1, emb2], dim=-1)          # (N, T, 2·d_gnn)

        out, _ = self.lstm(combined)                        # (N, T, 2·H)
        context = self.attention(out)                       # (N, 2·H)
        return self.norm(context)

    def forward(
        self,
        kp_seq_m1: torch.Tensor,
        kp_seq_m2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        kp_seq_m1 : Tensor (N, T, n_nodes, n_feat)  — mouse 1
        kp_seq_m2 : Tensor (N, T, n_nodes, n_feat)  — mouse 2

        Returns
        -------
        logits : Tensor (N, n_classes)
        """
        context = self.get_embedding(kp_seq_m1, kp_seq_m2)  # (N, 2·H)
        context = self.drop(context)
        return self.head(context)                            # (N, n_classes)

    @property
    def d_model(self) -> int:
        """Width of the LSTM embedding, for compatibility with TemporalAttention."""
        return self.lstm.hidden_size * 2


# ---------------------------------------------------------------------------
# Dataset wrapper for BehaviorSkeletonGNN
# ---------------------------------------------------------------------------

class SkeletonWindowDataset(torch.utils.data.Dataset):
    """Per-keypoint window dataset for ``BehaviorSkeletonGNN``.

    Parameters
    ----------
    kp_m1 : np.ndarray, shape (N, T, n_nodes, n_feat)
        Keypoint windows for mouse 1.
    kp_m2 : np.ndarray, shape (N, T, n_nodes, n_feat)
        Keypoint windows for mouse 2.
    y : np.ndarray, shape (N,)
        Integer class labels.
    """

    def __init__(
        self,
        kp_m1: "np.ndarray",
        kp_m2: "np.ndarray",
        y: "np.ndarray",
    ) -> None:
        import numpy as np
        mask = y >= 0
        self.kp_m1 = torch.from_numpy(kp_m1[mask].astype(np.float32))
        self.kp_m2 = torch.from_numpy(kp_m2[mask].astype(np.float32))
        self.y = torch.from_numpy(y[mask].astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.kp_m1[idx], self.kp_m2[idx], self.y[idx]
