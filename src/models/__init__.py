"""src/models — classifiers for mouse behaviour.

Provides the classical baselines and the BiLSTM, together with the later
architectures: the TCN, the Transformer and the skeleton GNN.
"""

from src.models.baseline import RandomForestBaseline, SVMBaseline, GradientBoostingBaseline
from src.models.rnn import (
    BehaviorLSTM,
    BehaviorCNNLSTM,
    BehaviorTCN,
    BehaviorTransformer,
    BehaviorGRU,
    FocalLoss,
)

__all__ = [
    "RandomForestBaseline",
    "SVMBaseline",
    "GradientBoostingBaseline",
    "BehaviorLSTM",
    "BehaviorCNNLSTM",
    "BehaviorTCN",
    "BehaviorTransformer",
    "BehaviorGRU",
    "FocalLoss",
]
