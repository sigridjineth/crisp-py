import torch

from .base import PruningStrategy


class TailPruning(PruningStrategy):
    """Takes the last k tokens from sequences"""

    def __init__(self, k_query: int, k_doc: int):
        if k_query <= 0:
            raise ValueError("k_query must be positive")
        if k_doc <= 0:
            raise ValueError("k_doc must be positive")
        self.k_query = k_query
        self.k_doc = k_doc

    def prune(
        self, embeddings: torch.Tensor, mask: torch.Tensor, is_query: bool
    ) -> torch.Tensor:
        k = self.k_query if is_query else self.k_doc
        batch_size, _, embed_dim = embeddings.shape
        device = embeddings.device

        result = []

        for i in range(batch_size):
            # get last k valid tokens
            valid_indices = torch.where(mask[i] > 0)[0]

            if len(valid_indices) == 0:
                result.append(torch.zeros(k, embed_dim, device=device))
            elif len(valid_indices) <= k:
                # pad if needed
                selected = embeddings[i, valid_indices]
                padded = torch.zeros(k, embed_dim, device=device)
                padded[: len(valid_indices)] = selected
                result.append(padded)
            else:
                selected_indices = valid_indices[-k:]
                result.append(embeddings[i, selected_indices])

        return torch.stack(result)

    def get_output_size(self, input_size: int, is_query: bool) -> int:
        return self.k_query if is_query else self.k_doc

    def __repr__(self) -> str:
        return f"TailPruning(k_query={self.k_query}, k_doc={self.k_doc})"
