"""
Unit tests for pruning strategies.
"""

import pytest
import torch
from crisp.pruning.spacing import KSpacing
from crisp.pruning.tail import TailPruning


class TestTailPruning:
    """Test TailPruning strategy."""

    def test_initialization(self):
        """Test TailPruning initialization."""
        pruner = TailPruning(k_query=4, k_doc=8)
        assert pruner.k_query == 4
        assert pruner.k_doc == 8

        # Test invalid values
        with pytest.raises(ValueError, match="k_query must be positive"):
            TailPruning(k_query=0, k_doc=8)

        with pytest.raises(ValueError, match="k_doc must be positive"):
            TailPruning(k_query=4, k_doc=-1)

    def test_prune_query(self, sample_embeddings, sample_mask):
        """Test pruning query embeddings."""
        k_query = 4
        pruner = TailPruning(k_query=k_query, k_doc=8)

        result = pruner.prune(sample_embeddings, sample_mask, is_query=True)

        # Check shape
        assert result.shape == (
            sample_embeddings.size(0),
            k_query,
            sample_embeddings.size(2),
        )

        # Check that we got the last k tokens
        for i in range(sample_embeddings.size(0)):
            valid_indices = torch.where(sample_mask[i] > 0)[0]
            if len(valid_indices) >= k_query:
                expected = sample_embeddings[i, valid_indices[-k_query:]]
                torch.testing.assert_close(result[i], expected)

    def test_prune_document(self, sample_embeddings, sample_mask):
        """Test pruning document embeddings."""
        k_doc = 8
        pruner = TailPruning(k_query=4, k_doc=k_doc)

        result = pruner.prune(sample_embeddings, sample_mask, is_query=False)

        # Check shape
        assert result.shape == (
            sample_embeddings.size(0),
            k_doc,
            sample_embeddings.size(2),
        )

    def test_prune_with_padding(self, batch_size, embed_dim, device):
        """Test pruning when sequence has fewer tokens than k."""
        k = 10
        seq_len = 5
        pruner = TailPruning(k_query=k, k_doc=k)

        # Create embeddings shorter than k
        embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device)
        mask = torch.ones(batch_size, seq_len, device=device)

        result = pruner.prune(embeddings, mask, is_query=True)

        # Check shape
        assert result.shape == (batch_size, k, embed_dim)

        # Check that first seq_len positions contain the embeddings
        for i in range(batch_size):
            torch.testing.assert_close(result[i, :seq_len], embeddings[i])
            # Check padding is zeros
            torch.testing.assert_close(
                result[i, seq_len:], torch.zeros(k - seq_len, embed_dim, device=device)
            )

    def test_prune_empty_sequence(self, batch_size, embed_dim, device):
        """Test pruning when sequence has no valid tokens."""
        k = 4
        seq_len = 10
        pruner = TailPruning(k_query=k, k_doc=k)

        embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device)
        mask = torch.zeros(batch_size, seq_len, device=device)  # All masked

        result = pruner.prune(embeddings, mask, is_query=True)

        # Should return all zeros
        expected = torch.zeros(batch_size, k, embed_dim, device=device)
        torch.testing.assert_close(result, expected)

    def test_get_output_size(self):
        """Test get_output_size method."""
        pruner = TailPruning(k_query=4, k_doc=8)

        assert pruner.get_output_size(100, is_query=True) == 4
        assert pruner.get_output_size(100, is_query=False) == 8
        assert pruner.get_output_size(2, is_query=True) == 4  # Always returns k

    def test_repr(self):
        """Test string representation."""
        pruner = TailPruning(k_query=4, k_doc=8)
        assert repr(pruner) == "TailPruning(k_query=4, k_doc=8)"


class TestKSpacing:
    """Test KSpacing strategy."""

    def test_initialization(self):
        """Test KSpacing initialization."""
        pruner = KSpacing(k=2)
        assert pruner.k == 2

        pruner = KSpacing(k=4)
        assert pruner.k == 4

        # Test invalid values
        with pytest.raises(ValueError, match="k must be at least 1"):
            KSpacing(k=0)

        with pytest.raises(ValueError, match="k must be at least 1"):
            KSpacing(k=-1)

    def test_prune_k2(self, sample_embeddings, sample_mask):
        """Test K=2 spacing (every 2nd token)."""
        pruner = KSpacing(k=2)

        result = pruner.prune(sample_embeddings, sample_mask, is_query=True)

        # Check shape - should have approximately half the tokens
        batch_size, seq_len, embed_dim = sample_embeddings.shape

        for i in range(batch_size):
            valid_indices = torch.where(sample_mask[i] > 0)[0]
            if len(valid_indices) > 0:
                # Select every 2nd token
                selected_indices = valid_indices[::2]
                expected_count = len(selected_indices)

                # Result should match expected selection
                assert result[i, :expected_count].shape[0] == expected_count

                # Check actual values
                for j, idx in enumerate(selected_indices):
                    torch.testing.assert_close(result[i, j], sample_embeddings[i, idx])

    def test_prune_k4(self, sample_embeddings, sample_mask):
        """Test K=4 spacing (every 4th token)."""
        pruner = KSpacing(k=4)

        result = pruner.prune(sample_embeddings, sample_mask, is_query=True)

        # Check that we got every 4th token
        for i in range(sample_embeddings.size(0)):
            valid_indices = torch.where(sample_mask[i] > 0)[0]
            if len(valid_indices) > 0:
                selected_indices = valid_indices[::4]
                expected_count = len(selected_indices)

                # Check first few tokens match
                for j in range(min(expected_count, 5)):
                    torch.testing.assert_close(
                        result[i, j], sample_embeddings[i, selected_indices[j]]
                    )

    def test_prune_single_token(self, batch_size, embed_dim, device):
        """Test pruning when there's only one valid token."""
        pruner = KSpacing(k=2)

        seq_len = 10
        embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device)
        mask = torch.zeros(batch_size, seq_len, device=device)
        mask[:, 0] = 1  # Only first token is valid

        result = pruner.prune(embeddings, mask, is_query=True)

        # Should select the one valid token
        for i in range(batch_size):
            torch.testing.assert_close(result[i, 0], embeddings[i, 0])

    def test_get_output_size(self, sample_mask):
        """Test get_output_size method."""
        pruner = KSpacing(k=2)

        # For k=2, output is approximately input_size // 2
        assert pruner.get_output_size(100, is_query=True) == 50
        assert pruner.get_output_size(101, is_query=True) == 51  # Rounds up

        pruner = KSpacing(k=4)
        assert pruner.get_output_size(100, is_query=True) == 25
        assert pruner.get_output_size(102, is_query=True) == 26  # Rounds up

    def test_repr(self):
        """Test string representation."""
        pruner = KSpacing(k=2)
        assert repr(pruner) == "KSpacing(k=2)"

        pruner = KSpacing(k=4)
        assert repr(pruner) == "KSpacing(k=4)"

    @pytest.mark.parametrize(
        "k,seq_len,expected_output",
        [
            (2, 10, 5),
            (2, 11, 6),
            (3, 9, 3),
            (3, 10, 4),
            (4, 16, 4),
            (4, 17, 5),
        ],
    )
    def test_output_size_calculations(self, k, seq_len, expected_output):
        """Test various output size calculations."""
        pruner = KSpacing(k=k)
        assert pruner.get_output_size(seq_len, is_query=True) == expected_output

    def test_consistency_between_runs(self, sample_embeddings, sample_mask):
        """Test that pruning is deterministic."""
        pruner = KSpacing(k=2)

        result1 = pruner.prune(sample_embeddings, sample_mask, is_query=True)
        result2 = pruner.prune(sample_embeddings, sample_mask, is_query=True)

        torch.testing.assert_close(result1, result2)
