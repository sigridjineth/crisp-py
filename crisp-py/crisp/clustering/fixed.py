import logging

import torch

from ..config import CRISPConfig
from .base import ClusteringStrategy
from .kmeans import KMeansClustering

logger = logging.getLogger(__name__)


class FixedClustering(ClusteringStrategy):
    """Fixed number of clusters (C4x8, C8x32)"""

    def __init__(self, k_query: int, k_doc: int, config: CRISPConfig):
        # Validation
        if k_query <= 0:
            raise ValueError("k_query must be positive")
        if k_doc <= 0:
            raise ValueError("k_doc must be positive")

        self.k_query = k_query
        self.k_doc = k_doc
        self.kmeans = KMeansClustering(
            n_iterations=config.kmeans_iterations,
            max_iterations=config.kmeans_max_iterations,
            use_faiss=config.use_faiss_clustering,
            log_stats=config.log_clustering_stats,
        )

        # log faiss fallback
        if hasattr(self.kmeans, "use_faiss") and not self.kmeans.use_faiss:
            logger.info(
                "FAISS not available, using PyTorch implementation for " "clustering"
            )

    def cluster(
        self, embeddings: torch.Tensor, mask: torch.Tensor, is_query: bool
    ) -> torch.Tensor:
        """
        Cluster embeddings to fixed number of centroids.

        Args:
            embeddings: Token embeddings of shape
                (batch_size, seq_len, embed_dim)
            mask: Attention mask of shape (batch_size, seq_len)
            is_query: Whether these are query embeddings

        Returns:
            Clustered embeddings of shape (batch_size, k, embed_dim)
            where k = k_query if is_query else k_doc
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

        k = self.k_query if is_query else self.k_doc
        batch_size = embeddings.size(0)
        device = embeddings.device
        embed_dim = embeddings.size(-1)

        result = []

        for i in range(batch_size):
            # Extract valid embeddings
            valid_mask = mask[i] > 0
            valid_embeddings = embeddings[i][valid_mask]

            if valid_embeddings.size(0) == 0:
                # No valid embeddings, return zeros
                result.append(torch.zeros(k, embed_dim, device=device))
            elif valid_embeddings.size(0) <= k:
                # Fewer embeddings than clusters, pad with zeros
                padded = torch.zeros(k, embed_dim, device=device)
                padded[: valid_embeddings.size(0)] = valid_embeddings
                result.append(padded)
            else:
                # Cluster and get centroids
                centroids = self.kmeans.cluster(valid_embeddings, k)
                result.append(centroids)

        return torch.stack(result)

    def get_num_clusters(self, num_tokens: int, is_query: bool) -> int:
        """Return the fixed number of clusters."""
        return self.k_query if is_query else self.k_doc

    def __repr__(self) -> str:
        """Return string representation."""
        return f"FixedClustering(k_query={self.k_query}, k_doc={self.k_doc})"
