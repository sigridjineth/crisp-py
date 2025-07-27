"""
Relative-size clustering strategy implementation.

This module implements clustering to a percentage of the sequence length
(C25, C50).
"""

import logging

import torch

from ..config import CRISPConfig
from .base import ClusteringStrategy
from .kmeans import KMeansClustering

logger = logging.getLogger(__name__)


class RelativeClustering(ClusteringStrategy):
    """
    Cluster embeddings to a percentage of sequence length.

    This strategy implements the C25 and C50 methods from the CRISP paper,
    which cluster to 25% or 50% of the original sequence length respectively.

    Attributes:
        percentage: Fraction of tokens to retain (0.25 for C25, 0.50 for C50)
        max_k: Maximum number of clusters (based on max_doc_length)
        kmeans: K-means clustering instance
    """

    def __init__(self, percentage: float, config: CRISPConfig):
        """
        Initialize relative clustering strategy.

        Args:
            percentage: Fraction of tokens to retain (e.g., 0.25 for 25%)
            config: CRISP configuration for K-means parameters

        Raises:
            ValueError: If percentage is not in (0, 1]
        """
        if not 0 < percentage <= 1:
            raise ValueError(f"percentage must be in (0, 1], got {percentage}")

        self.percentage = percentage
        self.max_k = int(config.max_doc_length * percentage)
        self.kmeans = KMeansClustering(
            n_iterations=config.kmeans_iterations,
            max_iterations=config.kmeans_max_iterations,
            use_faiss=config.use_faiss_clustering,
            log_stats=config.log_clustering_stats,
        )

        # Log FAISS availability
        if hasattr(self.kmeans, "use_faiss") and not self.kmeans.use_faiss:
            logger.info(
                "FAISS not available, using PyTorch implementation for " "clustering"
            )

    def cluster(
        self, embeddings: torch.Tensor, mask: torch.Tensor, is_query: bool
    ) -> torch.Tensor:
        """
        Cluster embeddings to a percentage of sequence length.

        Args:
            embeddings: Token embeddings of shape
                (batch_size, seq_len, embed_dim)
            mask: Attention mask of shape (batch_size, seq_len)
            is_query: Whether these are query embeddings (unused in relative
                clustering)

        Returns:
            Clustered embeddings of shape (batch_size, max_k, embed_dim)
            where actual clusters per sequence depend on valid token count
        """
        # Input validation
        if embeddings.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {embeddings.dim()}D")
        if mask.dim() != 2:
            raise ValueError(f"Expected 2D mask tensor, got {mask.dim()}D")
        if embeddings.size(0) != mask.size(0) or embeddings.size(1) != mask.size(1):
            raise ValueError(
                f"Embeddings shape {embeddings.shape} incompatible with "
                f"mask shape {mask.shape}"
            )

        batch_size = embeddings.size(0)
        device = embeddings.device
        embed_dim = embeddings.size(-1)

        # First pass: determine the maximum k across the batch
        max_k_in_batch = 0
        for i in range(batch_size):
            valid_mask = mask[i] > 0
            n_valid = valid_mask.sum().item()
            if n_valid > 0:
                k = max(1, int(n_valid * self.percentage))
                max_k_in_batch = max(max_k_in_batch, k)

        # If no valid embeddings in any sample, return minimal tensor
        if max_k_in_batch == 0:
            return torch.zeros(batch_size, 1, embed_dim, device=device)

        results = []

        for i in range(batch_size):
            # Extract valid embeddings
            valid_mask = mask[i] > 0
            valid_embeddings = embeddings[i][valid_mask]
            n_valid = valid_embeddings.size(0)

            if n_valid == 0:
                # No valid embeddings, pad with zeros
                results.append(torch.zeros(max_k_in_batch, embed_dim, device=device))
            else:
                # Calculate number of clusters based on percentage
                k = max(1, int(n_valid * self.percentage))

                if n_valid <= k:
                    # Fewer embeddings than desired clusters
                    centroids = valid_embeddings
                else:
                    # Perform clustering
                    centroids = self.kmeans.cluster(valid_embeddings, k)

                # Pad to max_k_in_batch for consistent batching
                padded = torch.zeros(max_k_in_batch, embed_dim, device=device)
                padded[: centroids.size(0)] = centroids
                results.append(padded)

        return torch.stack(results)

    def get_num_clusters(self, num_tokens: int, is_query: bool) -> int:
        """Calculate number of clusters based on percentage."""
        return max(1, min(int(num_tokens * self.percentage), self.max_k))

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RelativeClustering(percentage={self.percentage}, " f"max_k={self.max_k})"
        )
