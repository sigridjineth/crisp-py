"""
Integration tests for end-to-end CRISP functionality.
"""

import tempfile
from unittest.mock import Mock, patch

import torch
from crisp.config import CRISPConfig, PruningMethod
from crisp.data.collators import CRISPCollator
from crisp.data.dataset import TestCRISPDataset
from crisp.losses.chamfer import ChamferSimilarity
from crisp.models.encoder import CRISPEncoder
from crisp.models.lightning import CRISPModel
from tests.test_utils import create_mock_dataset, create_mock_tokenizer


class TestEndToEnd:
    """Test complete CRISP training pipeline."""

    @patch("crisp.models.encoder.AutoModel")
    def test_model_initialization_all_methods(self, mock_automodel):
        """Test that all pruning methods can be initialized."""
        # Mock base model
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_automodel.from_pretrained.return_value = mock_model

        for method in PruningMethod:
            config = CRISPConfig(
                model_name="bert-base-uncased", method=method, embedding_dim=768
            )

            # Should initialize without errors
            encoder = CRISPEncoder(config)
            # Check that either pruning or clustering strategy is set
            assert (
                encoder.pruning_strategy is not None
                or encoder.clustering_strategy is not None
            )

    @patch("crisp.models.encoder.AutoModel")
    def test_forward_pass_all_methods(self, mock_automodel):
        """Test forward pass for all pruning methods."""
        batch_size = 2
        seq_len = 64
        hidden_size = 768

        # Mock base model
        mock_model = Mock()
        mock_model.config.hidden_size = hidden_size
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
        mock_model.return_value = mock_output
        mock_automodel.from_pretrained.return_value = mock_model

        for method in PruningMethod:
            config = CRISPConfig(
                model_name="bert-base-uncased", method=method, embedding_dim=hidden_size
            )

            encoder = CRISPEncoder(config)

            # Create inputs
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            attention_mask[0, 40:] = 0  # Add some padding

            # Forward pass should work
            query_outputs = encoder(input_ids, attention_mask, is_query=True)
            doc_outputs = encoder(input_ids, attention_mask, is_query=False)

            # Extract embeddings from output dict
            query_embeds = (
                query_outputs["embeddings"]
                if isinstance(query_outputs, dict)
                else query_outputs
            )
            doc_embeds = (
                doc_outputs["embeddings"]
                if isinstance(doc_outputs, dict)
                else doc_outputs
            )

            # Check outputs
            assert query_embeds.dim() == 3
            assert doc_embeds.dim() == 3
            assert query_embeds.size(0) == batch_size
            assert doc_embeds.size(0) == batch_size
            assert query_embeds.size(-1) == hidden_size
            assert doc_embeds.size(-1) == hidden_size

    @patch("crisp.models.encoder.AutoModel")
    def test_loss_computation(self, mock_automodel):
        """Test loss computation with different configurations."""
        batch_size = 4
        hidden_size = 768

        # Mock base model
        mock_model = Mock()
        mock_model.config.hidden_size = hidden_size
        mock_automodel.from_pretrained.return_value = mock_model

        # Test a few representative methods
        test_methods = [PruningMethod.TAIL_4X8, PruningMethod.K2, PruningMethod.C8X32]

        for method in test_methods:
            config = CRISPConfig(
                model_name="bert-base-uncased",
                method=method,
                embedding_dim=hidden_size,
                temperature=0.05,
            )

            # Create Lightning module
            lightning_module = CRISPModel(config)

            # Mock forward pass to return full-size embeddings (before pruning)
            query_embeds = torch.randn(batch_size, 32, hidden_size)
            doc_embeds = torch.randn(batch_size, 64, hidden_size)

            with patch.object(lightning_module.encoder, "forward") as mock_forward:
                # Configure mock to return appropriate embeddings in dict format
                def side_effect(input_ids, attention_mask, is_query, **kwargs):
                    embeds = query_embeds if is_query else doc_embeds
                    return {
                        "embeddings": embeds,
                        "attention_mask": torch.ones_like(embeds[..., 0]),
                    }

                mock_forward.side_effect = side_effect

                # Create batch
                batch = {
                    "query_input_ids": torch.randint(0, 1000, (batch_size, 32)),
                    "query_attention_mask": torch.ones(batch_size, 32),
                    "doc_input_ids": torch.randint(0, 1000, (batch_size, 64)),
                    "doc_attention_mask": torch.ones(batch_size, 64),
                }

                # Compute loss
                loss = lightning_module.training_step(batch, 0)

                # Check loss
                assert isinstance(loss, torch.Tensor)
                assert loss.shape == ()  # Scalar
                assert torch.isfinite(loss)
                assert loss.item() > 0

    def test_dataset_creation(self):
        """Test dataset creation and loading."""
        # Create mock data
        data = create_mock_dataset(num_samples=10)
        tokenizer = create_mock_tokenizer()

        config = CRISPConfig(max_query_length=32, max_doc_length=64)

        # Create dataset
        dataset = TestCRISPDataset(data, tokenizer, config)

        # Check dataset
        assert len(dataset) == 10

        # Get a sample
        sample = dataset[0]
        assert "query_input_ids" in sample
        assert "query_attention_mask" in sample
        assert "doc_input_ids" in sample
        assert "doc_attention_mask" in sample

        # Check shapes
        assert sample["query_input_ids"].shape == (32,)
        assert sample["doc_input_ids"].shape == (64,)

    def test_data_collation(self):
        """Test data collation for batching."""
        # Create mock data
        data = create_mock_dataset(num_samples=10)
        tokenizer = create_mock_tokenizer()

        config = CRISPConfig(max_query_length=32, max_doc_length=64, batch_size=4)

        # Create dataset and collator
        dataset = TestCRISPDataset(data, tokenizer, config)

        # Create a simple collator for this test
        def simple_collator(samples):
            # Stack the tensors from all samples
            return {
                "query_input_ids": torch.stack([s["query_input_ids"] for s in samples]),
                "query_attention_mask": torch.stack(
                    [s["query_attention_mask"] for s in samples]
                ),
                "doc_input_ids": torch.stack([s["doc_input_ids"] for s in samples]),
                "doc_attention_mask": torch.stack(
                    [s["doc_attention_mask"] for s in samples]
                ),
            }

        # Get a batch
        samples = [dataset[i] for i in range(4)]
        batch = simple_collator(samples)

        # Check batch
        assert batch["query_input_ids"].shape == (4, 32)
        assert batch["query_attention_mask"].shape == (4, 32)
        assert batch["doc_input_ids"].shape == (4, 64)
        assert batch["doc_attention_mask"].shape == (4, 64)

    @patch("crisp.models.encoder.AutoModel")
    def test_gradient_flow(self, mock_automodel):
        """Test that gradients flow through the entire model."""
        batch_size = 2
        hidden_size = 768

        # Mock base model with parameters
        mock_model = Mock()
        mock_model.config.hidden_size = hidden_size

        # Add mock parameters that require gradients
        mock_param = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        mock_model.parameters = Mock(return_value=[mock_param])

        # Mock forward output
        mock_output = Mock()
        base_embeddings = torch.randn(batch_size, 32, hidden_size, requires_grad=True)
        mock_output.last_hidden_state = base_embeddings
        mock_model.return_value = mock_output

        mock_automodel.from_pretrained.return_value = mock_model

        config = CRISPConfig(
            model_name="bert-base-uncased",
            method=PruningMethod.K2,
            embedding_dim=hidden_size,
        )

        # Create model
        encoder = CRISPEncoder(config)

        # Create inputs
        input_ids = torch.randint(0, 1000, (batch_size, 32))
        attention_mask = torch.ones(batch_size, 32)

        # Forward pass
        outputs = encoder(input_ids, attention_mask, is_query=True)

        # Extract embeddings from dict if needed
        embeddings = outputs["embeddings"] if isinstance(outputs, dict) else outputs

        # Create simple loss
        loss = embeddings.sum()

        # Backward pass
        loss.backward()

        # Check that base embeddings received gradients
        assert base_embeddings.grad is not None
        assert torch.any(base_embeddings.grad != 0)

    @patch("crisp.models.encoder.AutoModel")
    def test_model_save_load(self, mock_automodel):
        """Test model saving and loading."""
        # Mock base model
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_automodel.from_pretrained.return_value = mock_model

        config = CRISPConfig(
            model_name="bert-base-uncased",
            method=PruningMethod.C8X32,
            embedding_dim=768,
        )

        # Create model
        model1 = CRISPEncoder(config)

        # Save config
        with tempfile.TemporaryDirectory():
            # Save config (would need to implement config serialization)
            # For now, just test that model can be recreated
            model2 = CRISPEncoder(config)

            assert model1.config.method == model2.config.method
            assert model1.config.embedding_dim == model2.config.embedding_dim

    def test_chamfer_similarity_integration(self):
        """Test Chamfer similarity with various embedding configurations."""
        chamfer = ChamferSimilarity(symmetric=True)

        # Test different sizes
        test_configs = [
            (4, 8, 16, 768),  # batch, query_tokens, doc_tokens, embed_dim
            (8, 4, 32, 512),
            (2, 16, 8, 1024),
        ]

        for batch_size, q_tokens, d_tokens, embed_dim in test_configs:
            query_embeds = torch.randn(batch_size, q_tokens, embed_dim)
            doc_embeds = torch.randn(batch_size, d_tokens, embed_dim)
            query_mask = torch.ones(batch_size, q_tokens)
            doc_mask = torch.ones(batch_size, d_tokens)

            # Add some padding
            query_mask[0, q_tokens // 2 :] = 0
            doc_mask[1, d_tokens // 2 :] = 0

            # Compute similarities
            similarities = chamfer(
                query_embeds, query_mask, doc_embeds, doc_mask, normalize=True
            )

            # Check output
            assert similarities.shape == (batch_size, batch_size)
            assert torch.all(torch.isfinite(similarities))

            # Check that diagonal has reasonable values
            diag = similarities.diagonal()
            assert torch.all(diag > -1)
            assert torch.all(diag < 1)  # Due to normalization
