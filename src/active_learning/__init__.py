"""
src/active_learning
====================
A semi-automatic labelling pipeline built on active learning.

Provides:
- ``query_strategies`` — the query strategies: entropy, BALD, coreset and others
- ``oracle``           — the simulated expert annotator
"""

from .oracle import SimulatedOracle
from .query_strategies import (
    QueryStrategy,
    LeastConfidenceStrategy,
    MarginSamplingStrategy,
    EntropySamplingStrategy,
    BALDStrategy,
    CoresetStrategy,
    CombinedStrategy,
    HybridCoresetEntropyStrategy,
)

__all__ = [
    "SimulatedOracle",
    "QueryStrategy",
    "LeastConfidenceStrategy",
    "MarginSamplingStrategy",
    "EntropySamplingStrategy",
    "BALDStrategy",
    "CoresetStrategy",
    "CombinedStrategy",
    "HybridCoresetEntropyStrategy",
]
