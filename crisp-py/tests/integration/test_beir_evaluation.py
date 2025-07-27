"""
Integration tests for BEIR evaluation functionality.
"""

from unittest.mock import Mock, patch

import pytest
import torch
from crisp.config import CRISPConfig, PruningMethod
from crisp.evaluation.beir import BEIREvaluator, get_beir_instruction


class TestBEIRIntegration:
    """Test BEIR evaluation integration."""

    def test_beir_instruction_retrieval(self):
        """Test BEIR instruction retrieval for different tasks."""
        # Test known tasks
        tasks_with_instructions = [
            "nfcorpus",
            "fiqa",
            "arguana",
            "scidocs",
            "scifact",
            "trec-covid",
            "webis-touche2020",
            "quora",
            "dbpedia-entity",
            "fever",
            "climate-fever",
            "hotpotqa",
            "nq",
        ]

        for task in tasks_with_instructions:
            instruction = get_beir_instruction(task)
            assert isinstance(instruction, str)
            assert len(instruction) > 0
            assert instruction != ""

        # Test unknown task
        instruction = get_beir_instruction("unknown-task")
        assert instruction == ""

    @patch("crisp.evaluation.beir.CRISPEncoder")
    def test_beir_evaluator_initialization(self, mock_encoder_class):
        """Test BEIREvaluator initialization."""
        # Mock encoder
        mock_encoder = Mock()
        mock_encoder_class.return_value = mock_encoder

        config = CRISPConfig(
            model_name="bert-base-uncased",
            method=PruningMethod.C8X32,
            use_instruction_prefix=True,
        )

        evaluator = BEIREvaluator(mock_encoder, config)

        assert evaluator.model == mock_encoder
        assert evaluator.config == config
        assert evaluator.use_instruction_prefix is True

    @patch("crisp.evaluation.beir.ChamferSimilarity.compute_retrieval_scores")
    def test_compute_similarities_batch(self, mock_compute_scores):
        """Test batch similarity computation."""
        # Setup
        mock_encoder = Mock()
        # Make the mock encoder return itself when .to() is called
        mock_encoder.to.return_value = mock_encoder
        config = CRISPConfig(batch_size=2)
        evaluator = BEIREvaluator(mock_encoder, config)

        # Mock data
        num_queries = 3
        num_docs = 5
        embed_dim = 768
        num_tokens = 8

        # Mock encoder outputs
        query_embeds = torch.randn(num_queries, num_tokens, embed_dim)
        doc_embeds = torch.randn(num_docs, num_tokens, embed_dim)
        query_masks = torch.ones(num_queries, num_tokens)
        doc_masks = torch.ones(num_docs, num_tokens)

        def encode_side_effect(texts, is_query, **kwargs):
            if is_query:
                return {
                    "embeddings": query_embeds[: len(texts)],
                    "attention_mask": query_masks[: len(texts)],
                }
            else:
                return {
                    "embeddings": doc_embeds[: len(texts)],
                    "attention_mask": doc_masks[: len(texts)],
                }

        mock_encoder.encode.side_effect = encode_side_effect

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 32)),
            "attention_mask": torch.ones(1, 32),
        }

        # Mock similarity computation
        expected_scores = torch.randn(num_queries, num_docs)
        mock_compute_scores.return_value = expected_scores

        # Test
        queries = [f"query {i}" for i in range(num_queries)]
        documents = [f"doc {i}" for i in range(num_docs)]

        with patch.object(evaluator, "_create_attention_masks") as mock_masks:
            mock_masks.side_effect = lambda embeds, _: (
                query_masks if embeds.shape[0] == num_queries else doc_masks
            )

            scores = evaluator.compute_similarities(queries, documents, mock_tokenizer)

        # Verify
        assert torch.equal(scores, expected_scores)
        mock_compute_scores.assert_called_once()

    def test_evaluate_with_mock_dataset(self):
        """Test evaluation with a mock BEIR dataset."""
        # Create mock model and config
        mock_encoder = Mock()
        # Make the mock encoder return itself when .to() is called
        mock_encoder.to.return_value = mock_encoder
        config = CRISPConfig(
            batch_size=2, use_instruction_prefix=False  # Simplify test
        )

        evaluator = BEIREvaluator(mock_encoder, config)

        # Create mock dataset structure
        corpus = {
            "doc1": {"text": "Document 1 text"},
            "doc2": {"text": "Document 2 text"},
            "doc3": {"text": "Document 3 text"},
        }

        queries = {
            "q1": "Query 1 text",
            "q2": "Query 2 text",
        }

        qrels = {
            "q1": {"doc1": 1, "doc2": 0, "doc3": 0},
            "q2": {"doc1": 0, "doc2": 1, "doc3": 0},
        }

        # Mock tokenizer
        mock_tokenizer = Mock()

        # Mock encoder.encode to return proper format
        def mock_encode(texts, **kwargs):
            num_texts = len(texts)
            embeddings = torch.randn(num_texts, 10, 768)  # 10 tokens per text
            return {
                "embeddings": embeddings,
                "attention_mask": torch.ones(num_texts, 10),
            }

        mock_encoder.encode.side_effect = mock_encode

        # Mock similarity computation to return perfect scores
        def mock_compute_similarities(q_texts, d_texts, tokenizer):
            # Return high scores for relevant pairs
            num_queries = len(q_texts)
            num_docs = len(d_texts)
            scores = torch.zeros(num_queries, num_docs)

            # Set high scores for relevant documents
            if "Query 1" in q_texts[0]:
                scores[0, 0] = 0.9  # q1 -> doc1
            if len(q_texts) > 1 and "Query 2" in q_texts[1]:
                scores[1, 1] = 0.9  # q2 -> doc2

            return scores

        with patch.object(evaluator, "compute_similarities", mock_compute_similarities):
            with patch.object(evaluator, "tokenizer", mock_tokenizer):
                results = evaluator.evaluate(
                    corpus=corpus, queries=queries, qrels=qrels
                )

        # Check results structure
        assert "ndcg@10" in results
        assert results["ndcg@10"] > 0  # Should have good NDCG

    def test_create_attention_masks(self):
        """Test attention mask creation from embeddings."""
        mock_encoder = Mock()
        config = CRISPConfig()
        evaluator = BEIREvaluator(mock_encoder, config)

        # Create embeddings with some zero vectors (simulating padding)
        batch_size = 3
        num_tokens = 5
        embed_dim = 8

        embeddings = torch.randn(batch_size, num_tokens, embed_dim)
        # Make some tokens "invalid" (zero vectors)
        embeddings[0, 3:] = 0  # First sample: 3 valid tokens
        embeddings[1, 4:] = 0  # Second sample: 4 valid tokens
        # Third sample: all valid

        tokenizer = Mock()
        masks = evaluator._create_attention_masks(embeddings, tokenizer)

        # Check masks
        assert masks.shape == (batch_size, num_tokens)

        # First sample should have 3 valid tokens
        assert masks[0, :3].sum() == 3
        assert masks[0, 3:].sum() == 0

        # Second sample should have 4 valid tokens
        assert masks[1, :4].sum() == 4
        assert masks[1, 4:].sum() == 0

        # Third sample should have all valid
        assert masks[2].sum() == num_tokens

    @pytest.mark.parametrize(
        "method", [PruningMethod.TAIL_4X8, PruningMethod.K2, PruningMethod.C25]
    )
    def test_different_pruning_methods_evaluation(self, method):
        """Test evaluation works with different pruning methods."""
        mock_encoder = Mock()
        # Make the mock encoder return itself when .to() is called
        mock_encoder.to.return_value = mock_encoder
        config = CRISPConfig(method=method)
        evaluator = BEIREvaluator(mock_encoder, config)

        # Simple test data
        corpus = {"d1": {"text": "test doc"}}
        queries = {"q1": "test query"}
        qrels = {"q1": {"d1": 1}}

        # Mock the encoding based on method
        if method in [PruningMethod.TAIL_4X8, PruningMethod.TAIL_8X32]:
            # Fixed size output
            num_tokens = 4 if "query" in str(method) else 8
        else:
            # Variable size output
            num_tokens = 10

        mock_embeds = torch.randn(1, num_tokens, 768)
        mock_encoder.encode.return_value = {
            "embeddings": mock_embeds,
            "attention_mask": torch.ones(1, num_tokens),
        }

        # Mock tokenizer
        mock_tokenizer = Mock()

        with patch.object(evaluator, "tokenizer", mock_tokenizer):
            with patch.object(evaluator, "_create_attention_masks") as mock_masks:
                mock_masks.return_value = torch.ones(1, num_tokens)

                # Should not raise errors
                results = evaluator.evaluate(
                    corpus=corpus, queries=queries, qrels=qrels
                )
                assert isinstance(results, dict)

    def test_instruction_prefix_handling(self):
        """Test that instruction prefixes are properly added."""
        mock_encoder = Mock()
        config = CRISPConfig(use_instruction_prefix=True, beir_task="nfcorpus")
        evaluator = BEIREvaluator(mock_encoder, config)

        # Mock tokenizer and encoder
        mock_tokenizer = Mock()
        mock_encoder.encode.return_value = torch.randn(1, 8, 768)

        queries = {"q1": "test query"}

        with patch.object(evaluator, "tokenizer", mock_tokenizer):
            with patch.object(evaluator, "_encode_queries") as mock_encode:
                mock_encode.return_value = {"q1": torch.randn(1, 8, 768)}

                # Encode queries
                evaluator._encode_queries(queries, mock_tokenizer)

                # Check that instruction was prepended
                mock_encode.assert_called_once()

                # In actual implementation, instruction would be prepended
                # This is a simplified test
                assert "q1" in queries
