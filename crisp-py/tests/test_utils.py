"""
Test utilities and helper functions for CRISP tests.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from crisp.config import PruningMethod


def create_mock_dataset(
    num_samples: int = 100,
    query_length: int = 32,
    doc_length: int = 128,
    vocab_size: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Create a mock dataset for testing.

    Args:
        num_samples: Number of samples to create
        query_length: Maximum query length
        doc_length: Maximum document length
        vocab_size: Vocabulary size for random tokens
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries with query and document data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = []
    for i in range(num_samples):
        # Variable lengths
        q_len = np.random.randint(10, query_length)
        d_len = np.random.randint(20, doc_length)

        dataset.append(
            {
                "query": f"Query text {i} " * (q_len // 5),
                "document": f"Document text {i} " * (d_len // 5),
                "query_id": f"q{i}",
                "doc_id": f"d{i}",
                "relevance": 1 if i % 3 == 0 else 0,  # 1/3 are relevant
            }
        )

    return dataset


def create_random_embeddings(
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    normalized: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create random embeddings for testing.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        embed_dim: Embedding dimension
        normalized: Whether to L2 normalize embeddings
        device: Device to create tensors on

    Returns:
        Random embeddings tensor
    """
    if device is None:
        device = torch.device("cpu")

    embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device)

    if normalized:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    return embeddings


def create_attention_mask(
    batch_size: int,
    seq_len: int,
    min_valid_ratio: float = 0.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create attention masks with variable padding.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        min_valid_ratio: Minimum ratio of valid tokens (non-padded)
        device: Device to create tensors on

    Returns:
        Attention mask tensor
    """
    if device is None:
        device = torch.device("cpu")

    mask = torch.ones(batch_size, seq_len, device=device)

    for i in range(batch_size):
        # Random valid length
        min_valid = int(seq_len * min_valid_ratio)
        valid_len = torch.randint(min_valid, seq_len + 1, (1,)).item()

        if valid_len < seq_len:
            mask[i, valid_len:] = 0

    return mask


def assert_embeddings_shape(
    embeddings: torch.Tensor,
    expected_batch_size: int,
    expected_num_tokens: Optional[int] = None,
    expected_embed_dim: int = None,
):
    """
    Assert embeddings have expected shape.

    Args:
        embeddings: Embeddings to check
        expected_batch_size: Expected batch size
        expected_num_tokens: Expected number of tokens (optional)
        expected_embed_dim: Expected embedding dimension
    """
    assert embeddings.dim() == 3, f"Expected 3D tensor, got {embeddings.dim()}D"
    assert (
        embeddings.size(0) == expected_batch_size
    ), f"Expected batch size {expected_batch_size}, got {embeddings.size(0)}"

    if expected_num_tokens is not None:
        assert (
            embeddings.size(1) == expected_num_tokens
        ), f"Expected {expected_num_tokens} tokens, got {embeddings.size(1)}"

    if expected_embed_dim is not None:
        assert (
            embeddings.size(2) == expected_embed_dim
        ), f"Expected embed dim {expected_embed_dim}, got {embeddings.size(2)}"


def get_expected_output_size(
    method: PruningMethod, input_size: int, is_query: bool
) -> int:
    """
    Get expected output size for a pruning method.

    Args:
        method: Pruning method
        input_size: Input sequence length
        is_query: Whether this is for queries

    Returns:
        Expected output size after pruning
    """
    if method == PruningMethod.TAIL_4X8:
        return 4 if is_query else 8
    elif method == PruningMethod.TAIL_8X32:
        return 8 if is_query else 32
    elif method == PruningMethod.K2:
        return (input_size + 1) // 2
    elif method == PruningMethod.K4:
        return (input_size + 3) // 4
    elif method == PruningMethod.C4X8:
        return 4 if is_query else 8
    elif method == PruningMethod.C8X32:
        return 8 if is_query else 32
    elif method == PruningMethod.C25:
        return max(1, int(input_size * 0.25))
    elif method == PruningMethod.C50:
        return max(1, int(input_size * 0.5))
    else:
        raise ValueError(f"Unknown method: {method}")


def create_similarity_matrix(
    batch_size: int,
    diagonal_value: float = 0.9,
    off_diagonal_range: Tuple[float, float] = (-0.2, 0.2),
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a similarity matrix for testing contrastive losses.

    Args:
        batch_size: Size of the matrix
        diagonal_value: Value for diagonal elements (positives)
        off_diagonal_range: Range for off-diagonal elements
        device: Device to create tensor on

    Returns:
        Similarity matrix
    """
    if device is None:
        device = torch.device("cpu")

    # Start with random values in specified range
    low, high = off_diagonal_range
    similarities = (
        torch.rand(batch_size, batch_size, device=device) * (high - low) + low
    )

    # Set diagonal to specified value
    similarities.fill_diagonal_(diagonal_value)

    return similarities


def compute_expected_loss(
    similarities: torch.Tensor, temperature: float = 0.05
) -> float:
    """
    Compute expected InfoNCE loss for a similarity matrix.

    Args:
        similarities: Similarity matrix
        temperature: Temperature parameter

    Returns:
        Expected loss value
    """
    # Scale by temperature
    similarities = similarities / temperature

    # Compute log softmax along each row
    log_probs = torch.log_softmax(similarities, dim=1)

    # Extract diagonal (correct) probabilities
    correct_log_probs = torch.diagonal(log_probs)

    # Average negative log probability
    loss = -correct_log_probs.mean()

    return loss.item()


def check_gradient_flow(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    loss_fn: Optional[torch.nn.Module] = None,
) -> bool:
    """
    Check if gradients flow properly through a model.

    Args:
        model: Model to test
        inputs: Input dictionary
        loss_fn: Optional loss function (uses sum if not provided)

    Returns:
        True if gradients flow properly
    """
    # Set model to training mode
    model.train()

    # Zero gradients
    model.zero_grad()

    # Forward pass
    outputs = model(**inputs)

    # Compute loss
    if loss_fn is not None:
        loss = loss_fn(outputs)
    else:
        loss = outputs.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    has_gradients = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            if torch.any(param.grad != 0):
                has_gradients = True
                # Also check for NaN/Inf
                if torch.any(torch.isnan(param.grad)) or torch.any(
                    torch.isinf(param.grad)
                ):
                    return False

    return has_gradients


def create_mock_tokenizer():
    """Create a mock tokenizer for testing."""

    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.cls_token_id = 101
            self.sep_token_id = 102

        def __call__(self, texts, **kwargs):
            # Simple mock tokenization
            if isinstance(texts, str):
                texts = [texts]

            batch_size = len(texts)
            max_length = kwargs.get("max_length", 128)

            input_ids = torch.randint(103, 1000, (batch_size, max_length))
            attention_mask = torch.ones(batch_size, max_length)

            # Add some padding
            for i in range(batch_size):
                pad_len = i * 10
                if pad_len < max_length:
                    input_ids[i, -pad_len:] = self.pad_token_id
                    attention_mask[i, -pad_len:] = 0

            return {"input_ids": input_ids, "attention_mask": attention_mask}

    return MockTokenizer()


def compare_embeddings(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Compare two embedding tensors for approximate equality.

    Args:
        embeddings1: First embedding tensor
        embeddings2: Second embedding tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if embeddings are approximately equal
    """
    return torch.allclose(embeddings1, embeddings2, rtol=rtol, atol=atol)
