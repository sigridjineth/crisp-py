"""
Pruning strategies for CRISP models.

This module contains various token pruning strategies including tail selection
and k-spacing methods.
"""

from typing import Optional

from ..config import CRISPConfig, PruningMethod
from .base import PruningStrategy
from .spacing import KSpacing
from .tail import TailPruning


def create_pruning_strategy(config: CRISPConfig) -> Optional[PruningStrategy]:
    """
    Create appropriate pruning strategy from config.

    Args:
        config: CRISP configuration containing method

    Returns:
        PruningStrategy instance or None if method is not a pruning method

    Note:
        This only handles fixed-token pruning methods (tail, k-spacing).
        For clustering methods, use create_clustering_strategy() instead.
    """
    method = config.method

    # Tail methods
    if method == PruningMethod.TAIL_4X8:
        return TailPruning(k_query=4, k_doc=8)
    elif method == PruningMethod.TAIL_8X32:
        return TailPruning(k_query=8, k_doc=32)

    # K-spacing methods
    elif method == PruningMethod.K2:
        return KSpacing(k=2)
    elif method == PruningMethod.K4:
        return KSpacing(k=4)

    # Not a pruning method (likely clustering)
    else:
        return None


__all__ = ["PruningStrategy", "TailPruning", "KSpacing", "create_pruning_strategy"]
