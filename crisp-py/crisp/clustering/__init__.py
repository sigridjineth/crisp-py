"""
Clustering strategies for CRISP models.

This module contains various clustering strategies including fixed-size
clustering and relative-size clustering using K-means.
"""

from typing import Union

from ..config import CRISPConfig, PruningMethod
from .base import ClusteringStrategy
from .fixed import FixedClustering
from .kmeans import KMeansClustering
from .relative import RelativeClustering


def create_clustering_strategy(config: CRISPConfig) -> Union[ClusteringStrategy, None]:
    """
    Create appropriate clustering strategy from config.

    Args:
        config: CRISP configuration

    Returns:
        ClusteringStrategy instance or None if method is not clustering-based
    """
    method = config.method

    # Fixed clustering methods
    if method == PruningMethod.C4X8:
        return FixedClustering(k_query=4, k_doc=8, config=config)
    elif method == PruningMethod.C8X32:
        return FixedClustering(k_query=8, k_doc=32, config=config)

    # Relative clustering methods
    elif method == PruningMethod.C25:
        return RelativeClustering(percentage=0.25, config=config)
    elif method == PruningMethod.C50:
        return RelativeClustering(percentage=0.50, config=config)

    # Not a clustering method
    else:
        return None


__all__ = [
    "ClusteringStrategy",
    "KMeansClustering",
    "FixedClustering",
    "RelativeClustering",
    "create_clustering_strategy",
]
