from typing import Tuple

import pytest
import torch
from crisp.config import CRISPConfig, PruningMethod


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def embed_dim():
    return 768


@pytest.fixture
def seq_len():
    return 128


@pytest.fixture
def default_config():
    return CRISPConfig(
        model_name="bert-base-uncased",
        embedding_dim=768,
        max_query_length=128,
        max_doc_length=256,
        batch_size=4,
        kmeans_iterations=5,
        kmeans_max_iterations=10,
        use_faiss_clustering=False,
        mixed_precision=False,
        num_workers=0,
    )


@pytest.fixture
def sample_embeddings(
    batch_size: int, seq_len: int, embed_dim: int, device: torch.device
) -> torch.Tensor:
    """Create sample embeddings for testing."""
    return torch.randn(batch_size, seq_len, embed_dim, device=device)


@pytest.fixture
def sample_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Create sample attention mask with some padding."""
    mask = torch.ones(batch_size, seq_len, device=device)
    # Add some padding at the end
    for i in range(batch_size):
        pad_len = i * 10  # Variable padding
        if pad_len > 0 and pad_len < seq_len:
            mask[i, -pad_len:] = 0
    return mask


@pytest.fixture
def query_doc_embeddings(
    batch_size: int, embed_dim: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample query and document embeddings with masks."""
    query_len = 32
    doc_len = 64

    query_embeds = torch.randn(batch_size, query_len, embed_dim, device=device)
    doc_embeds = torch.randn(batch_size, doc_len, embed_dim, device=device)

    # Create masks with variable lengths
    query_mask = torch.ones(batch_size, query_len, device=device)
    doc_mask = torch.ones(batch_size, doc_len, device=device)

    for i in range(batch_size):
        # Variable query lengths
        query_pad = i * 5
        if query_pad > 0 and query_pad < query_len:
            query_mask[i, -query_pad:] = 0

        # Variable doc lengths
        doc_pad = i * 10
        if doc_pad > 0 and doc_pad < doc_len:
            doc_mask[i, -doc_pad:] = 0

    return query_embeds, query_mask, doc_embeds, doc_mask


@pytest.fixture(params=list(PruningMethod))
def pruning_method(request):
    """Parametrized fixture for all pruning methods."""
    return request.param


@pytest.fixture
def normalized_embeddings(sample_embeddings: torch.Tensor) -> torch.Tensor:
    """Create L2-normalized embeddings."""
    return torch.nn.functional.normalize(sample_embeddings, p=2, dim=-1)
