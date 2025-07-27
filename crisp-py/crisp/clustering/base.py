"""
Base class for clustering strategies.

This module defines the abstract interface that all clustering strategies
must implement.
"""

from abc import ABC, abstractmethod

import torch


class ClusteringStrategy(ABC):
    """
    Abstract base class for all clustering strategies.

    Clustering strategies reduce the number of token embeddings by grouping
    similar tokens and representing each group with a single centroid.
    """

    @abstractmethod
    def cluster(
        self, embeddings: torch.Tensor, mask: torch.Tensor, is_query: bool
    ) -> torch.Tensor:
        """
        Apply clustering to token embeddings.

        Args:
            embeddings: Token embeddings of shape
                (batch_size, seq_len, embed_dim)
            mask: Attention mask of shape (batch_size, seq_len)
                indicating valid tokens
            is_query: Whether these are query embeddings (True) or
                document embeddings (False)

        Returns:
            Clustered embeddings of shape (batch_size, num_clusters, embed_dim)

        Note:
            The output shape's second dimension (num_clusters) depends on the
            specific clustering strategy and may differ between queries and
            documents.
        """
        pass

    def get_num_clusters(self, num_tokens: int, is_query: bool) -> int:
        """
        Get the number of clusters for a given number of tokens.

        Args:
            num_tokens: Number of input tokens
            is_query: Whether these are query tokens

        Returns:
            Number of clusters to create

        Note:
            This is an optional method that helps determine cluster count.
            Default implementation returns -1 to indicate dynamic sizing.
        """
        return -1

    def __repr__(self) -> str:
        """Return string representation of the clustering strategy."""
        return f"{self.__class__.__name__}()"
