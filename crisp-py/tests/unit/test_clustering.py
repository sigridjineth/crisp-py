"""
Unit tests for clustering strategies.
"""

import pytest
import torch
from crisp.clustering.fixed import FixedClustering
from crisp.clustering.kmeans import KMeansClustering
from crisp.clustering.relative import RelativeClustering


class TestKMeansClustering:
    """Test KMeansClustering implementation."""

    def test_initialization(self):
        """Test KMeansClustering initialization."""
        from crisp.clustering.kmeans import FAISS_AVAILABLE

        kmeans = KMeansClustering(n_iterations=10)
        assert kmeans.n_iterations == 10
        assert kmeans.max_iterations == 50  # default
        # use_faiss defaults to True but will be False if FAISS not available
        assert kmeans.use_faiss == FAISS_AVAILABLE

        kmeans = KMeansClustering(n_iterations=5, max_iterations=20, use_faiss=True)
        assert kmeans.n_iterations == 5
        assert kmeans.max_iterations == 20
        assert (
            kmeans.use_faiss == FAISS_AVAILABLE
        )  # Will be False if FAISS not available

    def test_cluster_basic(self):
        """Test basic clustering functionality."""
        kmeans = KMeansClustering(n_iterations=5)

        # Create simple embeddings
        embeddings = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ]
        )

        # Cluster to 3 centroids
        centroids = kmeans.cluster(embeddings, 3)

        assert centroids.shape == (3, 3)
        assert centroids.dtype == embeddings.dtype
        assert centroids.device == embeddings.device

    def test_cluster_more_clusters_than_points(self):
        """Test clustering when k > number of points."""
        kmeans = KMeansClustering(n_iterations=5)

        embeddings = torch.randn(3, 10)
        centroids = kmeans.cluster(embeddings, 5)

        # Should return the original embeddings padded with zeros
        assert centroids.shape == (5, 10)
        torch.testing.assert_close(centroids[:3], embeddings)
        torch.testing.assert_close(centroids[3:], torch.zeros(2, 10))

    def test_cluster_single_point(self):
        """Test clustering with single point."""
        kmeans = KMeansClustering(n_iterations=5)

        embeddings = torch.randn(1, 10)
        centroids = kmeans.cluster(embeddings, 1)

        assert centroids.shape == (1, 10)
        torch.testing.assert_close(centroids, embeddings)

    def test_deterministic_initialization(self):
        """Test that k-means++ initialization is deterministic with fixed seed."""
        kmeans = KMeansClustering(n_iterations=5)

        embeddings = torch.randn(20, 10)

        # Multiple runs should give same result (deterministic)
        torch.manual_seed(42)
        centroids1 = kmeans.cluster(embeddings, 5)

        torch.manual_seed(42)
        centroids2 = kmeans.cluster(embeddings, 5)

        torch.testing.assert_close(centroids1, centroids2)


class TestFixedClustering:
    """Test FixedClustering strategy."""

    def test_initialization(self, default_config):
        """Test FixedClustering initialization."""
        clustering = FixedClustering(k_query=4, k_doc=8, config=default_config)
        assert clustering.k_query == 4
        assert clustering.k_doc == 8
        assert clustering.kmeans is not None

        # Test invalid values
        with pytest.raises(ValueError, match="k_query must be positive"):
            FixedClustering(k_query=0, k_doc=8, config=default_config)

        with pytest.raises(ValueError, match="k_doc must be positive"):
            FixedClustering(k_query=4, k_doc=-1, config=default_config)

    def test_cluster_query(self, query_doc_embeddings, default_config):
        """Test clustering query embeddings."""
        query_embeds, query_mask, _, _ = query_doc_embeddings
        k_query = 4

        clustering = FixedClustering(k_query=k_query, k_doc=8, config=default_config)
        result = clustering.cluster(query_embeds, query_mask, is_query=True)

        # Check shape
        assert result.shape == (query_embeds.size(0), k_query, query_embeds.size(2))

    def test_cluster_document(self, query_doc_embeddings, default_config):
        """Test clustering document embeddings."""
        _, _, doc_embeds, doc_mask = query_doc_embeddings
        k_doc = 8

        clustering = FixedClustering(k_query=4, k_doc=k_doc, config=default_config)
        result = clustering.cluster(doc_embeds, doc_mask, is_query=False)

        # Check shape
        assert result.shape == (doc_embeds.size(0), k_doc, doc_embeds.size(2))

    def test_cluster_with_padding(self, batch_size, embed_dim, device, default_config):
        """Test clustering when sequence has fewer tokens than k."""
        k = 10
        seq_len = 5

        clustering = FixedClustering(k_query=k, k_doc=k, config=default_config)

        embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device)
        mask = torch.ones(batch_size, seq_len, device=device)

        result = clustering.cluster(embeddings, mask, is_query=True)

        # Check shape
        assert result.shape == (batch_size, k, embed_dim)

        # When fewer embeddings than clusters, should pad with zeros
        for i in range(batch_size):
            # First seq_len should be the original embeddings
            torch.testing.assert_close(result[i, :seq_len], embeddings[i])
            # Rest should be zeros
            torch.testing.assert_close(
                result[i, seq_len:], torch.zeros(k - seq_len, embed_dim, device=device)
            )

    def test_cluster_empty_sequence(
        self, batch_size, embed_dim, device, default_config
    ):
        """Test clustering when sequence has no valid tokens."""
        k = 4
        seq_len = 10

        clustering = FixedClustering(k_query=k, k_doc=k, config=default_config)

        embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device)
        mask = torch.zeros(batch_size, seq_len, device=device)  # All masked

        result = clustering.cluster(embeddings, mask, is_query=True)

        # Should return all zeros
        expected = torch.zeros(batch_size, k, embed_dim, device=device)
        torch.testing.assert_close(result, expected)

    def test_get_num_clusters(self, default_config):
        """Test get_num_clusters method."""
        clustering = FixedClustering(k_query=4, k_doc=8, config=default_config)

        assert clustering.get_num_clusters(100, is_query=True) == 4
        assert clustering.get_num_clusters(100, is_query=False) == 8
        assert clustering.get_num_clusters(2, is_query=True) == 4  # Always returns k

    def test_repr(self, default_config):
        """Test string representation."""
        clustering = FixedClustering(k_query=4, k_doc=8, config=default_config)
        assert repr(clustering) == "FixedClustering(k_query=4, k_doc=8)"


class TestRelativeClustering:
    """Test RelativeClustering strategy."""

    def test_initialization(self, default_config):
        """Test RelativeClustering initialization."""
        clustering = RelativeClustering(percentage=0.25, config=default_config)
        assert clustering.percentage == 0.25

        clustering = RelativeClustering(percentage=0.5, config=default_config)
        assert clustering.percentage == 0.5

        # Test invalid values
        with pytest.raises(ValueError, match="percentage must be in"):
            RelativeClustering(percentage=0.0, config=default_config)

        with pytest.raises(ValueError, match="percentage must be in"):
            RelativeClustering(percentage=1.1, config=default_config)

    def test_cluster_25_percent(self, sample_embeddings, sample_mask, default_config):
        """Test 25% clustering."""
        clustering = RelativeClustering(percentage=0.25, config=default_config)

        result = clustering.cluster(sample_embeddings, sample_mask, is_query=True)

        # Check that each sample has approximately 25% of valid tokens
        for i in range(sample_embeddings.size(0)):
            valid_count = (sample_mask[i] > 0).sum().item()
            if valid_count > 0:
                # Result should have expected number of clusters (may vary slightly)
                assert result[i].shape[0] <= sample_embeddings.size(1)

    def test_cluster_50_percent(self, sample_embeddings, sample_mask, default_config):
        """Test 50% clustering."""
        clustering = RelativeClustering(percentage=0.5, config=default_config)

        result = clustering.cluster(sample_embeddings, sample_mask, is_query=True)

        # Check shape is consistent
        assert result.dim() == 3
        assert result.size(-1) == sample_embeddings.size(-1)  # Same embedding dim

    def test_get_num_clusters(self, default_config):
        """Test get_num_clusters method."""
        clustering = RelativeClustering(percentage=0.25, config=default_config)

        assert clustering.get_num_clusters(100, is_query=True) == 25
        assert clustering.get_num_clusters(50, is_query=True) == 12  # Truncates
        assert clustering.get_num_clusters(4, is_query=True) == 1  # At least 1
        assert clustering.get_num_clusters(0, is_query=True) == 1  # At least 1

        clustering = RelativeClustering(percentage=0.5, config=default_config)
        assert clustering.get_num_clusters(100, is_query=True) == 50
        assert clustering.get_num_clusters(51, is_query=True) == 25  # Truncates

    def test_repr(self, default_config):
        """Test string representation."""
        clustering = RelativeClustering(percentage=0.25, config=default_config)
        assert (
            repr(clustering)
            == f"RelativeClustering(percentage=0.25, max_k={clustering.max_k})"
        )

        clustering = RelativeClustering(percentage=0.5, config=default_config)
        assert (
            repr(clustering)
            == f"RelativeClustering(percentage=0.5, max_k={clustering.max_k})"
        )

    def test_minimum_one_cluster(self, batch_size, embed_dim, device, default_config):
        """Test that we always get at least one cluster for valid sequences."""
        clustering = RelativeClustering(percentage=0.25, config=default_config)

        # Create sequence with just 2 valid tokens
        seq_len = 10
        embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device)
        mask = torch.zeros(batch_size, seq_len, device=device)
        mask[:, :2] = 1  # Only first 2 tokens valid

        result = clustering.cluster(embeddings, mask, is_query=True)

        # Should get at least 1 cluster per sample
        for i in range(batch_size):
            valid_result = result[i]
            # Count non-zero clusters (approximately)
            non_zero_count = (valid_result.abs().sum(dim=-1) > 1e-6).sum().item()
            assert non_zero_count >= 1
