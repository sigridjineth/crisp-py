"""
K-spacing pruning strategy implementation.

This module implements the k-spacing strategy which selects every k-th token
from the sequence.
"""

import torch

from .base import PruningStrategy


class KSpacing(PruningStrategy):
    """
    Select every k-th token from the sequence.

    This strategy implements the K2 and K4 methods from the CRISP paper,
    which select every 2nd or 4th valid token respectively.

    Attributes:
        k: Spacing interval (e.g., k=2 selects every 2nd token)
    """

    def __init__(self, k: int):
        """
        Initialize k-spacing strategy.

        Args:
            k: Spacing interval (must be positive)

        Raises:
            ValueError: If k is not positive
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k

    def prune(
        self, embeddings: torch.Tensor, mask: torch.Tensor, is_query: bool
    ) -> torch.Tensor:
        """
        Select every k-th valid token.

        Args:
            embeddings: Token embeddings of shape
                (batch_size, seq_len, embed_dim)
            mask: Attention mask of shape (batch_size, seq_len)
            is_query: Whether these are query embeddings (unused in this
                strategy)

        Returns:
            Selected embeddings of shape (batch_size, num_selected, embed_dim)

        Note:
            The output is padded to ensure consistent batch dimensions.
            num_selected = seq_len // k
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

        batch_size, seq_len, embed_dim = embeddings.shape
        device = embeddings.device

        # Calculate maximum number of selected tokens
        max_selected = (
            seq_len + self.k - 1
        ) // self.k  # Equivalent to ceil(seq_len / k)

        result = []
        result_mask = []

        for i in range(batch_size):
            # Find valid token indices
            valid_indices = torch.where(mask[i] > 0)[0]

            if len(valid_indices) == 0:
                # No valid tokens
                result.append(torch.zeros(max_selected, embed_dim, device=device))
                result_mask.append(torch.zeros(max_selected, device=device))
            else:
                # Select every k-th valid token
                selected_indices = valid_indices[:: self.k]
                selected = embeddings[i, selected_indices]

                # Pad if needed
                if len(selected_indices) < max_selected:
                    padded = torch.zeros(max_selected, embed_dim, device=device)
                    padded[: len(selected_indices)] = selected
                    result.append(padded)

                    mask_padded = torch.zeros(max_selected, device=device)
                    mask_padded[: len(selected_indices)] = 1
                    result_mask.append(mask_padded)
                else:
                    # Truncate to max_selected if we have more
                    result.append(selected[:max_selected])
                    result_mask.append(torch.ones(max_selected, device=device))

        return torch.stack(result)

    def get_output_size(self, input_size: int, is_query: bool) -> int:
        """Return the expected output size after k-spacing."""
        return (input_size + self.k - 1) // self.k  # Equivalent to ceil(input_size / k)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"KSpacing(k={self.k})"
