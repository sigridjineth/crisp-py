"""
Unit tests for loss functions.
"""

import pytest
import torch
from crisp.losses.chamfer import ChamferSimilarity
from crisp.losses.contrastive import InfoNCELoss


class TestChamferSimilarity:
    """Test ChamferSimilarity loss."""

    def test_initialization(self):
        """Test ChamferSimilarity initialization."""
        chamfer = ChamferSimilarity(symmetric=True)
        assert chamfer.symmetric is True

        chamfer = ChamferSimilarity(symmetric=False)
        assert chamfer.symmetric is False

    def test_forward_symmetric(self, query_doc_embeddings):
        """Test symmetric Chamfer similarity computation."""
        query_embeds, query_mask, doc_embeds, doc_mask = query_doc_embeddings

        chamfer = ChamferSimilarity(symmetric=True)
        similarities = chamfer(query_embeds, query_mask, doc_embeds, doc_mask)

        # Check shape - should be batch_size x batch_size
        batch_size = query_embeds.size(0)
        assert similarities.shape == (batch_size, batch_size)

        # Check symmetry properties
        # Diagonal should have highest values (same query-doc pairs)
        diag = similarities.diagonal()
        assert torch.all(
            diag >= 0
        )  # Similarities should be non-negative after normalization

    def test_forward_asymmetric(self, query_doc_embeddings):
        """Test asymmetric Chamfer similarity computation."""
        query_embeds, query_mask, doc_embeds, doc_mask = query_doc_embeddings

        chamfer = ChamferSimilarity(symmetric=False)
        similarities = chamfer(query_embeds, query_mask, doc_embeds, doc_mask)

        # Check shape - should be batch_size (diagonal only)
        batch_size = query_embeds.size(0)
        assert similarities.shape == (batch_size,)

    def test_normalization_effect(self, query_doc_embeddings):
        """Test effect of normalization on similarities."""
        query_embeds, query_mask, doc_embeds, doc_mask = query_doc_embeddings

        chamfer = ChamferSimilarity(symmetric=True)

        # With normalization
        sim_normalized = chamfer(
            query_embeds, query_mask, doc_embeds, doc_mask, normalize=True
        )

        # Without normalization
        sim_unnormalized = chamfer(
            query_embeds, query_mask, doc_embeds, doc_mask, normalize=False
        )

        # Shapes should be the same
        assert sim_normalized.shape == sim_unnormalized.shape

        # Values will be different
        assert not torch.allclose(sim_normalized, sim_unnormalized, atol=1e-3)

    def test_masking_behavior(self, batch_size, embed_dim, device):
        """Test that masking works correctly."""
        chamfer = ChamferSimilarity(symmetric=False)

        # Create embeddings where half are masked
        seq_len = 10
        query_embeds = torch.ones(batch_size, seq_len, embed_dim, device=device)
        doc_embeds = torch.ones(batch_size, seq_len, embed_dim, device=device)

        # Mask second half of queries and docs
        query_mask = torch.ones(batch_size, seq_len, device=device)
        query_mask[:, seq_len // 2 :] = 0
        doc_mask = torch.ones(batch_size, seq_len, device=device)
        doc_mask[:, seq_len // 2 :] = 0

        similarities = chamfer(
            query_embeds, query_mask, doc_embeds, doc_mask, normalize=False
        )

        # All similarities should be positive (all ones dot product)
        assert torch.all(similarities > 0)

    def test_empty_sequences(self, batch_size, embed_dim, device):
        """Test behavior with completely masked sequences."""
        chamfer = ChamferSimilarity(symmetric=False)

        seq_len = 10
        query_embeds = torch.randn(batch_size, seq_len, embed_dim, device=device)
        doc_embeds = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # All masked
        query_mask = torch.zeros(batch_size, seq_len, device=device)
        doc_mask = torch.zeros(batch_size, seq_len, device=device)

        similarities = chamfer(query_embeds, query_mask, doc_embeds, doc_mask)

        # Should handle gracefully - likely zeros or very small values
        assert similarities.shape == (batch_size,)
        assert torch.all(torch.isfinite(similarities))

    def test_compute_retrieval_scores(self, device):
        """Test static method for retrieval scoring."""
        num_queries = 3
        num_docs = 5
        num_tokens = 10
        embed_dim = 8

        query_embeds = torch.randn(num_queries, num_tokens, embed_dim, device=device)
        query_mask = torch.ones(num_queries, num_tokens, device=device)
        doc_embeds = torch.randn(num_docs, num_tokens, embed_dim, device=device)
        doc_mask = torch.ones(num_docs, num_tokens, device=device)

        scores = ChamferSimilarity.compute_retrieval_scores(
            query_embeds, query_mask, doc_embeds, doc_mask, batch_size=2
        )

        assert scores.shape == (num_queries, num_docs)
        assert torch.all(torch.isfinite(scores))


class TestInfoNCELoss:
    """Test InfoNCELoss implementation."""

    def test_initialization(self):
        """Test InfoNCELoss initialization."""
        loss_fn = InfoNCELoss(temperature=0.05)
        assert loss_fn.temperature == 0.05

        loss_fn = InfoNCELoss(temperature=0.1)
        assert loss_fn.temperature == 0.1

        # Test invalid temperature
        with pytest.raises(ValueError, match="Temperature must be positive"):
            InfoNCELoss(temperature=0.0)

        with pytest.raises(ValueError, match="Temperature must be positive"):
            InfoNCELoss(temperature=-0.1)

    def test_forward_basic(self):
        """Test basic InfoNCE loss computation."""
        batch_size = 4
        loss_fn = InfoNCELoss(temperature=0.05)

        # Create similarity matrix with clear positives on diagonal
        similarities = (
            torch.eye(batch_size) * 0.9 + torch.randn(batch_size, batch_size) * 0.1
        )

        loss = loss_fn(similarities)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss.item() > 0  # Loss should be positive
        assert torch.isfinite(loss)

    def test_forward_with_perfect_similarities(self):
        """Test loss with perfect diagonal similarities."""
        batch_size = 4
        loss_fn = InfoNCELoss(temperature=0.05)

        # Perfect similarities: diagonal = 1, others = 0
        similarities = torch.eye(batch_size)

        loss = loss_fn(similarities)

        # Loss should be relatively low for perfect predictions
        assert loss.item() < 1.0

    def test_forward_with_random_similarities(self):
        """Test loss with random similarities."""
        batch_size = 8
        loss_fn = InfoNCELoss(temperature=0.05)

        # Random similarities
        similarities = torch.randn(batch_size, batch_size)

        loss = loss_fn(similarities)

        # Loss should be higher for random similarities
        assert loss.item() > 1.0
        assert torch.isfinite(loss)

    def test_temperature_effect(self):
        """Test effect of temperature on loss."""
        batch_size = 4
        similarities = (
            torch.eye(batch_size) * 0.8 + torch.randn(batch_size, batch_size) * 0.1
        )

        # Lower temperature = sharper distribution = potentially lower loss for good similarities
        loss_low_temp = InfoNCELoss(temperature=0.01)(similarities)
        loss_high_temp = InfoNCELoss(temperature=1.0)(similarities)

        # Both should be valid
        assert torch.isfinite(loss_low_temp)
        assert torch.isfinite(loss_high_temp)

        # Values will be different
        assert abs(loss_low_temp.item() - loss_high_temp.item()) > 0.1

    def test_gradient_flow(self):
        """Test that gradients flow properly through the loss."""
        batch_size = 4
        loss_fn = InfoNCELoss(temperature=0.05)

        # Create similarities that require grad
        similarities = torch.randn(batch_size, batch_size, requires_grad=True)

        loss = loss_fn(similarities)
        loss.backward()

        # Check gradients exist and are finite
        assert similarities.grad is not None
        assert torch.all(torch.isfinite(similarities.grad))
        assert similarities.grad.shape == similarities.shape

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        batch_size = 4
        loss_fn = InfoNCELoss(temperature=0.05)

        # Test with very large similarities
        similarities = torch.eye(batch_size) * 100
        loss = loss_fn(similarities)
        assert torch.isfinite(loss)

        # Test with very small similarities
        similarities = (
            torch.eye(batch_size) * 0.001 + torch.randn(batch_size, batch_size) * 0.0001
        )
        loss = loss_fn(similarities)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16, 32])
    def test_different_batch_sizes(self, batch_size):
        """Test loss computation with different batch sizes."""
        torch.manual_seed(42)  # For reproducibility
        loss_fn = InfoNCELoss(temperature=0.05)

        # Create similarities with controlled noise to avoid edge cases
        similarities = torch.eye(batch_size) * 0.7
        # Add smaller noise to off-diagonal elements only
        noise = torch.randn(batch_size, batch_size) * 0.1
        noise.fill_diagonal_(0)
        similarities = similarities + noise

        loss = loss_fn(similarities)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() > 0
