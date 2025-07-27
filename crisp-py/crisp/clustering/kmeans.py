"""
K-means clustering implementation.

This module provides K-means clustering with convergence validation and
robust handling, supporting both FAISS (if available) and PyTorch
implementations.
"""

import logging

import numpy as np
import torch

# Optional FAISS import
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class KMeansClustering:
    """
    K-means clustering with convergence validation and robust handling.

    This class provides both FAISS-based (for efficiency) and PyTorch-based
    (for compatibility) implementations of K-means clustering.

    Attributes:
        n_iterations: Standard number of iterations to run
        max_iterations: Maximum iterations with convergence checking
        use_faiss: Whether to use FAISS implementation if available
        log_stats: Whether to log clustering statistics
    """

    def __init__(
        self,
        n_iterations: int = 20,
        max_iterations: int = 50,
        use_faiss: bool = True,
        log_stats: bool = False,
    ):
        """
        Initialize K-means clustering.

        Args:
            n_iterations: Number of iterations for standard k-means
            max_iterations: Maximum iterations with convergence checking
            use_faiss: Whether to use FAISS if available
            log_stats: Whether to log clustering statistics
        """
        self.n_iterations = n_iterations
        self.max_iterations = max_iterations
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.log_stats = log_stats

        if use_faiss and not FAISS_AVAILABLE:
            logger.warning(
                "FAISS requested but not available. Using PyTorch " "implementation."
            )

    def cluster(self, embeddings: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """
        Perform K-means clustering with convergence checking.

        Args:
            embeddings: Embeddings to cluster of shape (n_points, embed_dim)
            n_clusters: Number of clusters to create

        Returns:
            Cluster centroids of shape (n_clusters, embed_dim)

        Note:
            If there are fewer points than clusters, returns all points.
        """
        if embeddings.size(0) <= n_clusters:
            # Fewer points than clusters, return all points padded with zeros
            device = embeddings.device
            dtype = embeddings.dtype
            result = torch.zeros(
                n_clusters, embeddings.size(1), device=device, dtype=dtype
            )
            result[: embeddings.size(0)] = embeddings
            return result

        # FAISS requires minimum 39 * n_clusters points for training
        # Fall back to PyTorch if too few points
        if self.use_faiss and embeddings.size(0) >= 39 * n_clusters:
            return self._faiss_kmeans(embeddings, n_clusters)
        else:
            return self._pytorch_kmeans_with_convergence(embeddings, n_clusters)

    def _faiss_kmeans(self, embeddings: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """FAISS implementation for efficiency."""
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_np)

        # Configure K-means
        kmeans = faiss.Kmeans(
            embeddings_np.shape[1],
            n_clusters,
            niter=self.n_iterations,
            nredo=3,
            gpu=embeddings.is_cuda and faiss.get_num_gpus() > 0,
        )

        # Train
        kmeans.train(embeddings_np)

        # Get centroids
        centroids = torch.from_numpy(kmeans.centroids).to(embeddings.device)

        if self.log_stats:
            _, labels = kmeans.index.search(embeddings_np, 1)
            self._log_cluster_stats(labels.squeeze(), n_clusters)

        return centroids

    def _pytorch_kmeans_with_convergence(
        self, embeddings: torch.Tensor, n_clusters: int
    ) -> torch.Tensor:
        """PyTorch implementation with convergence checking and empty cluster
        handling."""
        n_points, dim = embeddings.shape

        # Initialize centers using k-means++
        centers = self._kmeans_plus_plus_init(embeddings, n_clusters)

        prev_assignments = None
        converged = False

        for iteration in range(self.max_iterations):
            # Assignment step
            distances = torch.cdist(embeddings, centers)
            assignments = distances.argmin(dim=1)

            # Check convergence
            if prev_assignments is not None:
                if torch.equal(assignments, prev_assignments):
                    converged = True
                    if self.log_stats:
                        logger.info(
                            f"K-means converged after {iteration + 1} " f"iterations"
                        )
                    break

            prev_assignments = assignments.clone()

            # Update centers with empty cluster handling
            for i in range(n_clusters):
                cluster_mask = assignments == i
                if cluster_mask.sum() > 0:
                    centers[i] = embeddings[cluster_mask].mean(dim=0)
                else:
                    # Reinitialize empty cluster to random point
                    centers[i] = embeddings[torch.randint(n_points, (1,))]
                    if self.log_stats:
                        logger.warning(f"Reinitialized empty cluster {i}")

            # Early stopping if iterations exceed threshold
            if iteration >= self.n_iterations and iteration < self.max_iterations:
                # Check if making progress
                if iteration % 5 == 0:
                    changes = (assignments != prev_assignments).sum().item()
                    if changes < n_points * 0.01:  # Less than 1% change
                        break

        if not converged and self.log_stats:
            logger.warning(
                f"K-means did not fully converge after "
                f"{self.max_iterations} iterations"
            )

        if self.log_stats:
            self._log_cluster_stats(assignments.cpu().numpy(), n_clusters)

        return centers

    def _kmeans_plus_plus_init(
        self, embeddings: torch.Tensor, n_clusters: int
    ) -> torch.Tensor:
        """K-means++ initialization for better convergence."""
        n_points = embeddings.size(0)
        centers = []

        # First center is random
        centers.append(embeddings[torch.randint(n_points, (1,))])

        for _ in range(1, n_clusters):
            # Compute distances to nearest center
            dists = torch.stack(
                [torch.cdist(embeddings, c.unsqueeze(0)).squeeze() for c in centers]
            ).min(dim=0)[0]

            # Sample proportional to squared distance
            probs = dists.pow(2)
            probs = probs / probs.sum()

            # Handle numerical issues
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                # Fall back to uniform sampling
                idx = torch.randint(n_points, (1,))
            else:
                idx = torch.multinomial(probs, 1)

            centers.append(embeddings[idx])

        return torch.cat(centers, dim=0)

    def _log_cluster_stats(self, assignments: np.ndarray, n_clusters: int):
        """Log clustering statistics for debugging."""
        unique, counts = np.unique(assignments, return_counts=True)
        logger.info(
            f"Cluster sizes: min={counts.min()}, max={counts.max()}, "
            f"mean={counts.mean():.1f}, empty={n_clusters - len(unique)}"
        )
