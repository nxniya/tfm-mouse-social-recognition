"""
src/models/rnn.py
=================
Temporal architectures for behaviour classification.

Provides:
- ``FocalLoss``          — focal loss for class imbalance, with tunable gamma
- ``BehaviorLSTM``       — two-layer bidirectional LSTM, the baseline
- ``BehaviorCNNLSTM``    — residual CNN followed by a BiLSTM
- ``BehaviorTCN``        — temporal convolution network, dilated causal convs
- ``BehaviorTransformer``— Transformer encoder with sinusoidal positional encoding

Input shape: (N, T, F) = (batch, window_size=64, n_features=50)
Output shape: (N, n_classes)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for imbalanced multi-class classification.

    Down-weights the easy, high-confidence examples so that training
    concentrates on the hard ones, which here are the rare classes.

    .. math::
        FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing factor. 0 reduces to plain cross-entropy; 2 is the value
        recommended by Lin et al. (2017).
    weight : Tensor | None
        Per-class weights, shape ``(n_classes,)``. Accepts the weights returned
        by ``MouseBehaviorDataset.class_weights()`` directly.
    reduction : str
        ``"mean"`` | ``"sum"`` | ``"none"``.
    label_smoothing : float
        Label smoothing, as a regulariser. 0 disables it.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor (N, C) — raw logits, with NO softmax applied
        targets : Tensor (N,)  — class labels, int64
        """
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal_loss = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# BiLSTM
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Temporal Attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Pooling by learned attention over the time dimension.

    Computes a scalar score for each time step and returns the weighted sum of
    the hidden states, giving a global context vector for the sequence.

    .. math::
        e_t = \\tanh(W h_t + b)
        \\alpha_t = \\text{softmax}(e_t)
        c = \\sum_t \\alpha_t h_t

    Parameters
    ----------
    d_model : int
        Size of the input vector; 2 * hidden_size for a BiLSTM.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : Tensor (N, T, d_model)

        Returns
        -------
        context : Tensor (N, d_model)
        """
        scores = self.score(hidden_states)          # (N, T, 1)
        weights = torch.softmax(scores, dim=1)      # (N, T, 1)
        return (weights * hidden_states).sum(dim=1)  # (N, d_model)


class BehaviorLSTM(nn.Module):
    """Two-layer bidirectional LSTM with temporal attention.

    Consumes a whole window (T=64 frames, F features) and produces one set of
    class logits per window.

    Architecture::

        (N, T, F)
            ↓
        BiLSTM × num_layers
            ↓
        TemporalAttention  <- weighted sum over every time step
            ↓
        LayerNorm + Dropout
            ↓
        Linear(2·hidden_size → n_classes)
            ↓
        logits (N, n_classes)

    The attention lets the model weight different moments of the window, for
    instance the speed peak in an attack, instead of relying solely on the final
    hidden state, which compresses the whole window into one vector.

    Parameters
    ----------
    input_size : int
        Number of features per frame.
    hidden_size : int
        LSTM units per direction; a BiLSTM therefore outputs 2 * hidden_size.
    num_layers : int
        Stacked layers.
    n_classes : int
        Number of behaviour classes.
    dropout : float
        Dropout between LSTM layers, which only applies when num_layers > 1,
        and before the final layer.
    bidirectional : bool
        When True, recommended, concatenate the forward and backward states.
    use_attention : bool
        When True, the default, use ``TemporalAttention``. When False, use
        global mean-pooling over T, which is more stable on a very small
        dataset, where the attention weights themselves start to overfit.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 8,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        d_out = hidden_size * (2 if bidirectional else 1)
        self.attention = TemporalAttention(d_out) if use_attention else None
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_out, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget gate bias = 1 (mejora gradientes a largo plazo)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled representation, taken before the linear head.

        Used by ``CoresetStrategy`` to measure distances in the learned
        representation space rather than in raw feature space.

        Parameters
        ----------
        x : Tensor (N, T, F)

        Returns
        -------
        embedding : Tensor (N, d_model)
        """
        outputs, _ = self.lstm(x)
        if self.use_attention:
            context = self.attention(outputs)   # (N, d_out)
        else:
            context = outputs.mean(dim=1)        # (N, d_out) — mean-pooling global
        return self.norm(context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, T, F)

        Returns
        -------
        logits : Tensor (N, n_classes)
        """
        context = self.get_embedding(x)    # (N, 2H)
        context = self.drop(context)
        return self.head(context)          # (N, n_classes)

    @property
    def d_model(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)


# ---------------------------------------------------------------------------
# CNN-LSTM
# ---------------------------------------------------------------------------

class _ResidualBlock1D(nn.Module):
    """Conv1D residual block: Conv, GroupNorm, GELU, Dropout, Conv, GroupNorm, skip.

    GroupNorm rather than BatchNorm1d, for three reasons:
    - it keeps no running_mean or running_var, so a NaN in the input features
      cannot poison the statistics for every subsequent batch
    - it behaves identically in train and eval, which is what made val_loss
      stable here
    - it is better suited to small batches (under 64) and short sequences

    Both the temporal length (padding = kernel // 2) and the channel count are
    preserved.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        # num_groups must divide channels; 8 groups is the usual choice for C >= 32
        n_groups = min(8, channels)
        while channels % n_groups != 0:
            n_groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GroupNorm(n_groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GroupNorm(n_groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, C, T)"""
        return self.act(x + self.block(x))


class BehaviorCNNLSTM(nn.Module):
    """CNN-BiLSTM with temporal attention.

    The CNN encoder picks up local motion patterns at several scales, such as
    short bursts and gestures, while the BiLSTM captures the global temporal
    dependency. The combination beats the plain BiLSTM on both accuracy and
    robustness to partial occlusion.

    Architecture::

        (N, T, F)
            ↓
        InputProjection   Linear(F → C) + LayerNorm
            ↓
        CNN encoder       3 × ResidualBlock1D (kernels 7→5→3, canales C)
            ↓
        BiLSTM × lstm_layers
            ↓
        TemporalAttention  <- global context for the sequence
            ↓
        LayerNorm + Dropout
            ↓
        Linear(2H → n_classes)
            ↓
        logits (N, n_classes)

    Parameters
    ----------
    input_size : int
        Number of features per frame (F).
    cnn_channels : int
        Channels in the CNN encoder (C).
    cnn_layers : int
        Number of Conv1D residual blocks.
    lstm_hidden : int
        LSTM units per direction.
    lstm_layers : int
        Stacked BiLSTM layers.
    n_classes : int
        Number of behaviour classes.
    dropout : float
        Global dropout, applied at half strength inside the CNN and at full
        strength in the head.
    bidirectional : bool
        When True, recommended, use a bidirectional LSTM.
    """

    def __init__(
        self,
        input_size: int,
        cnn_channels: int = 128,
        cnn_layers: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        n_classes: int = 8,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, cnn_channels),
            nn.LayerNorm(cnn_channels),
        )

        # CNN encoder: residual blocks with decreasing kernel sizes
        cnn_kernels = [7, 5, 3] + [3] * max(0, cnn_layers - 3)
        self.cnn_encoder = nn.Sequential(
            *[
                _ResidualBlock1D(cnn_channels, kernel_size=cnn_kernels[i], dropout=dropout * 0.5)
                for i in range(cnn_layers)
            ]
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        d_out = lstm_hidden * (2 if bidirectional else 1)
        self.attention = TemporalAttention(d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_out, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate bias = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, T, F)

        Returns
        -------
        logits : Tensor (N, n_classes)
        """
        x = self.input_proj(x)           # (N, T, C)
        x = x.transpose(1, 2)           # (N, C, T)
        x = self.cnn_encoder(x)         # (N, C, T)
        x = x.transpose(1, 2)           # (N, T, C)
        outputs, _ = self.lstm(x)       # (N, T, 2H)
        context = self.attention(outputs)  # (N, 2H)
        context = self.norm(context)
        context = self.drop(context)
        return self.head(context)        # (N, n_classes)

    @property
    def d_model(self) -> int:
        return self.lstm.hidden_size * (2 if self.lstm.bidirectional else 1)


# ---------------------------------------------------------------------------
# Temporal Convolution Network (TCN)
# ---------------------------------------------------------------------------

class _CausalDilatedBlock(nn.Module):
    """TCN block: two CausalConv1d, GroupNorm, GELU, Dropout stages, plus a skip.

    Dilated causal convolutions give a long temporal context without letting any
    future frame influence the present one. The residual projection is only
    inserted when the input and output channel counts differ.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size : int  — typically 3 or 5
    dilation : int     — dilation factor, 2^i for the i-th block
    dropout : float
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Causal padding: (kernel-1)*dilation on the left, none on the right
        self.pad = (kernel_size - 1) * dilation
        n_groups = min(8, out_channels)
        while out_channels % n_groups != 0:
            n_groups -= 1

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )
        self.norm1 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, C, T)"""
        # Causal padding: pad on the left only
        out = F.pad(x, (self.pad, 0))
        out = self.act(self.norm1(self.conv1(out)))
        out = self.drop(out)
        out = F.pad(out, (self.pad, 0))
        out = self.norm2(self.conv2(out))
        return self.act(out + self.skip(x))


class BehaviorTCN(nn.Module):
    """Temporal Convolution Network for behaviour classification.

    Dilated causal convolutions (dilation = 2^i) capture patterns at several
    temporal scales without recurrence. Faster than an LSTM on a GPU, since the
    whole sequence is processed in parallel rather than step by step.

    Architecture::

        (N, T, F)
            ↓
        InputProjection   Linear(F → d_model) + LayerNorm
            ↓
        TCN encoder       n_blocks × CausalDilatedBlock
                          dilations: 1, 2, 4, …, 2^(n_blocks-1)
            ↓
        TemporalAttention  ← suma ponderada sobre T
            ↓
        LayerNorm + Dropout
            ↓
        Linear(d_model → n_classes)
            ↓
        logits (N, n_classes)

    Parameters
    ----------
    input_size : int
        Number of features per frame (F).
    d_model : int
        Internal channel count of the TCN.
    n_blocks : int
        Number of dilated blocks. The effective receptive field is
        ``1 + 2 * (2^n_blocks - 1) * (kernel_size - 1)``.
    kernel_size : int
        Convolution kernel size.
    n_classes : int
        Number of behaviour classes.
    dropout : float
        Dropout inside the blocks and before the head.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_blocks: int = 5,
        kernel_size: int = 3,
        n_classes: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self._d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

        blocks = []
        for i in range(n_blocks):
            dilation = 2 ** i
            blocks.append(
                _CausalDilatedBlock(d_model, d_model, kernel_size, dilation, dropout)
            )
        self.tcn = nn.Sequential(*blocks)

        self.attention = TemporalAttention(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-head embedding, for CoresetStrategy.

        Parameters
        ----------
        x : Tensor (N, T, F)

        Returns
        -------
        embedding : Tensor (N, d_model)
        """
        h = self.input_proj(x)          # (N, T, d_model)
        h = h.transpose(1, 2)           # (N, d_model, T)
        h = self.tcn(h)                 # (N, d_model, T)
        h = h.transpose(1, 2)           # (N, T, d_model)
        return self.norm(self.attention(h))  # (N, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, T, F)

        Returns
        -------
        logits : Tensor (N, n_classes)
        """
        emb = self.get_embedding(x)     # (N, d_model)
        return self.head(self.drop(emb))

    @property
    def d_model(self) -> int:
        return self._d_model


# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------

class _SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017).

    Adds temporal position information with no learnable parameters, which is
    what makes it robust to a change in sequence length.

    Parameters
    ----------
    d_model : int
    max_len : int  — longest sequence supported
    dropout : float
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pos = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
        dim = torch.arange(0, d_model, 2).float()                  # (d_model/2,)
        div = torch.exp(-dim * (torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, T, d_model)"""
        return self.drop(x + self.pe[:, : x.size(1)])


class BehaviorTransformer(nn.Module):
    """Transformer encoder for behaviour classification.

    Self-attention relates any pair of time steps directly, capturing the long
    dependencies that recurrence tends to lose.

    Architecture::

        (N, T, F)
            ↓
        InputProjection   Linear(F → d_model) + LayerNorm
            ↓
        SinusoidalPositionalEncoding
            ↓
        TransformerEncoder   n_layers × (MHA + FFN + LayerNorm)
            ↓
        TemporalAttention    <- weighted sum, better than [CLS] at small T
            ↓
        LayerNorm + Dropout
            ↓
        Linear(d_model → n_classes)
            ↓
        logits (N, n_classes)

    Parameters
    ----------
    input_size : int
        Number of features per frame (F).
    d_model : int
        Width of the Transformer's internal space. Must be divisible by
        ``n_heads``.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of TransformerEncoderLayer layers.
    dim_feedforward : int
        Width of the inner feed-forward layer.
    n_classes : int
        Number of behaviour classes.
    dropout : float
        Dropout inside TransformerEncoderLayer and before the head.
    max_len : int
        Longest sequence the positional encoding supports.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        n_classes: int = 8,
        dropout: float = 0.2,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self._d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = _SinusoidalPositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN, which is more stable on small datasets
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.attention = TemporalAttention(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-head embedding, for CoresetStrategy.

        Parameters
        ----------
        x : Tensor (N, T, F)

        Returns
        -------
        embedding : Tensor (N, d_model)
        """
        h = self.input_proj(x)      # (N, T, d_model)
        h = self.pos_enc(h)         # (N, T, d_model)
        h = self.encoder(h)         # (N, T, d_model)
        return self.norm(self.attention(h))  # (N, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, T, F)

        Returns
        -------
        logits : Tensor (N, n_classes)
        """
        emb = self.get_embedding(x)
        return self.head(self.drop(emb))

    @property
    def d_model(self) -> int:
        return self._d_model


# ---------------------------------------------------------------------------
# Lightweight GRU, the reduced model used for the capacity ablation
# ---------------------------------------------------------------------------

class BehaviorGRU(nn.Module):
    """Single-layer unidirectional GRU with temporal attention.

    A drastically simpler architecture than the BiLSTM: one unidirectional GRU
    layer with ``hidden_size=64``, about 30k parameters against the BiLSTM's
    631k. It exists to quantify what the extra architectural complexity actually
    buys in a low-data regime.

    Architecture::

        (N, T, F)
            ↓
        GRU, 1 unidirectional layer
            ↓
        TemporalAttention
            ↓
        LayerNorm + Dropout
            ↓
        Linear(hidden_size → n_classes)
            ↓
        logits (N, n_classes)

    Parameters
    ----------
    input_size : int
        Number of features per frame.
    hidden_size : int
        GRU units. Defaults to 64, keeping the model under 100k parameters.
    n_classes : int
        Number of behaviour classes.
    dropout : float
        Dropout before the classification head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        n_classes: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.attention = TemporalAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-head embedding, for CoresetStrategy.

        Parameters
        ----------
        x : Tensor (N, T, F)

        Returns
        -------
        embedding : Tensor (N, hidden_size)
        """
        outputs, _ = self.gru(x)          # (N, T, H)
        context = self.attention(outputs)  # (N, H)
        return self.norm(context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, T, F)

        Returns
        -------
        logits : Tensor (N, n_classes)
        """
        context = self.get_embedding(x)
        return self.head(self.drop(context))

    @property
    def d_model(self) -> int:
        return self.hidden_size
