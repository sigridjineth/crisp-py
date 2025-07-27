"""
Unit tests for evaluation metrics.
"""

import pytest
import torch
from crisp.evaluation.metrics import NDCG, compute_mrr, compute_recall_at_k


class TestNDCG:
    """Test NDCG metric implementation."""

    def test_initialization(self):
        """Test NDCG initialization."""
        ndcg = NDCG(k=10)
        assert ndcg.k == 10

        ndcg = NDCG(k=5)
        assert ndcg.k == 5

    def test_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        ndcg = NDCG(k=5)

        # Perfect ranking: scores decrease monotonically
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1])
        relevance = torch.tensor([1, 1, 1, 1, 1, 0, 0])  # First 5 are relevant

        score = ndcg(scores, relevance)

        # Perfect ranking should give NDCG close to 1
        assert 0.9 < score <= 1.0

    def test_worst_ranking(self):
        """Test NDCG with worst possible ranking."""
        ndcg = NDCG(k=5)

        # Worst ranking: all relevant items have lowest scores
        scores = torch.tensor([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([0, 0, 0, 0, 0, 1, 1])  # Last 2 are relevant

        score = ndcg(scores, relevance)

        # Bad ranking should give low NDCG
        assert score < 0.5

    def test_no_relevant_items(self):
        """Test NDCG when no items are relevant."""
        ndcg = NDCG(k=5)

        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([0, 0, 0, 0, 0])  # Nothing relevant

        score = ndcg(scores, relevance)

        # No relevant items should give 0
        assert score == 0.0

    def test_all_relevant_items(self):
        """Test NDCG when all items are relevant."""
        ndcg = NDCG(k=5)

        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([1, 1, 1, 1, 1])  # All relevant

        score = ndcg(scores, relevance)

        # All relevant with perfect order should give 1.0
        assert score == 1.0

    def test_graded_relevance(self):
        """Test NDCG with graded relevance scores."""
        ndcg = NDCG(k=5)

        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
        relevance = torch.tensor([3, 2, 2, 1, 0, 3])  # Graded relevance

        score = ndcg(scores, relevance)

        # Should be between 0 and 1
        assert 0.0 <= score <= 1.0

    def test_k_truncation(self):
        """Test that NDCG only considers top-k items."""
        ndcg = NDCG(k=3)

        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([1, 1, 1, 1, 1])  # All relevant

        score = ndcg(scores, relevance)

        # Should only consider top 3 items
        assert score == 1.0  # Perfect for top-3

    def test_tie_handling(self):
        """Test NDCG with tied scores."""
        ndcg = NDCG(k=5)

        # Tied scores
        scores = torch.tensor([5.0, 5.0, 3.0, 3.0, 1.0])
        relevance = torch.tensor([1, 0, 1, 0, 1])

        score = ndcg(scores, relevance)

        # Should handle ties consistently
        assert 0.0 <= score <= 1.0

    def test_batch_computation(self):
        """Test NDCG computation for batch of queries."""
        ndcg = NDCG(k=3)

        # Batch of 2 queries
        scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        relevance = torch.tensor(
            [[1, 1, 0, 0, 0], [0, 0, 0, 1, 1]]  # First 2 relevant  # Last 2 relevant
        )

        # Compute for each query
        scores_list = []
        for i in range(scores.size(0)):
            score = ndcg(scores[i], relevance[i])
            scores_list.append(score)

        # First query should have high NDCG, second should have low
        assert scores_list[0] > 0.8
        assert scores_list[1] < 0.5


class TestMRR:
    """Test Mean Reciprocal Rank computation."""

    def test_first_position_relevant(self):
        """Test MRR when first item is relevant."""
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([1, 0, 0, 0, 0])

        mrr = compute_mrr(scores, relevance)

        # First position relevant = MRR of 1.0
        assert mrr == 1.0

    def test_second_position_relevant(self):
        """Test MRR when second item is relevant."""
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([0, 1, 0, 0, 0])

        mrr = compute_mrr(scores, relevance)

        # Second position relevant = MRR of 0.5
        assert mrr == 0.5

    def test_no_relevant_items(self):
        """Test MRR when no items are relevant."""
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([0, 0, 0, 0, 0])

        mrr = compute_mrr(scores, relevance)

        # No relevant items = MRR of 0
        assert mrr == 0.0

    def test_multiple_relevant_items(self):
        """Test MRR with multiple relevant items (only first counts)."""
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([0, 1, 1, 1, 0])

        mrr = compute_mrr(scores, relevance)

        # Only first relevant position counts = 1/2
        assert mrr == 0.5

    def test_last_position_relevant(self):
        """Test MRR when last item is relevant."""
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([0, 0, 0, 0, 1])

        mrr = compute_mrr(scores, relevance)

        # Last position (5th) relevant = MRR of 1/5
        assert abs(mrr - 0.2) < 1e-6


class TestRecallAtK:
    """Test Recall@K computation."""

    def test_perfect_recall(self):
        """Test recall when all relevant items are in top-k."""
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([1, 1, 0, 0, 0])  # 2 relevant items

        recall = compute_recall_at_k(scores, relevance, k=2)

        # Both relevant items in top-2 = recall of 1.0
        assert recall == 1.0

    def test_partial_recall(self):
        """Test recall when some relevant items are in top-k."""
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([1, 0, 1, 0, 1])  # 3 relevant items

        recall = compute_recall_at_k(scores, relevance, k=2)

        # Only 1 of 3 relevant items in top-2 = recall of 1/3
        assert abs(recall - 1 / 3) < 1e-6

    def test_zero_recall(self):
        """Test recall when no relevant items are in top-k."""
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([0, 0, 0, 1, 1])  # Last 2 are relevant

        recall = compute_recall_at_k(scores, relevance, k=2)

        # No relevant items in top-2 = recall of 0
        assert recall == 0.0

    def test_no_relevant_items(self):
        """Test recall when there are no relevant items."""
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([0, 0, 0, 0, 0])  # Nothing relevant

        recall = compute_recall_at_k(scores, relevance, k=3)

        # No relevant items = recall of 0 (not undefined)
        assert recall == 0.0

    def test_k_larger_than_list(self):
        """Test recall when k is larger than the number of items."""
        scores = torch.tensor([5.0, 4.0, 3.0])
        relevance = torch.tensor([0, 1, 1])  # 2 relevant items

        recall = compute_recall_at_k(scores, relevance, k=10)

        # All items considered = recall of 1.0
        assert recall == 1.0

    @pytest.mark.parametrize(
        "k,expected_recall",
        [
            (1, 1 / 3),  # 1 of 3 relevant in top-1
            (2, 1 / 3),  # 1 of 3 relevant in top-2
            (3, 2 / 3),  # 2 of 3 relevant in top-3
            (4, 2 / 3),  # 2 of 3 relevant in top-4
            (5, 3 / 3),  # 3 of 3 relevant in top-5
        ],
    )
    def test_various_k_values(self, k, expected_recall):
        """Test recall with various k values."""
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = torch.tensor([1, 0, 1, 0, 1])  # Positions 0, 2, 4 are relevant

        recall = compute_recall_at_k(scores, relevance, k=k)

        assert abs(recall - expected_recall) < 1e-6
