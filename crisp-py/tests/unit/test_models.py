"""
Unit tests for CRISP models.
"""

from unittest.mock import Mock, patch

import torch
from crisp.config import CRISPConfig, PruningMethod
from crisp.models.encoder import CRISPEncoder


class TestCRISPEncoder:
    """Test CRISPEncoder model."""

    @patch("crisp.models.encoder.AutoTokenizer")
    @patch("crisp.models.encoder.AutoModel")
    def test_initialization(self, mock_automodel, mock_autotokenizer, default_config):
        """Test CRISPEncoder initialization."""
        # Mock the base model
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_automodel.from_pretrained.return_value = mock_model

        # Mock the tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_autotokenizer.from_pretrained.return_value = mock_tokenizer

        encoder = CRISPEncoder(default_config)

        # Check that model was loaded
        mock_automodel.from_pretrained.assert_called_once()
        assert encoder.config == default_config
        assert encoder.encoder == mock_model

    @patch("crisp.models.encoder.AutoTokenizer")
    @patch("crisp.models.encoder.AutoModel")
    def test_initialization_with_different_methods(
        self, mock_automodel, mock_autotokenizer
    ):
        """Test initialization with different pruning methods."""
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_automodel.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_autotokenizer.from_pretrained.return_value = mock_tokenizer

        # Test TAIL methods
        config = CRISPConfig(
            method=PruningMethod.TAIL_4X8, model_name="bert-base-uncased"
        )
        encoder = CRISPEncoder(config)
        assert encoder.config.method == PruningMethod.TAIL_4X8

        # Test K-spacing methods
        config = CRISPConfig(method=PruningMethod.K2, model_name="bert-base-uncased")
        encoder = CRISPEncoder(config)
        assert encoder.config.method == PruningMethod.K2

        # Test clustering methods
        config = CRISPConfig(method=PruningMethod.C25, model_name="bert-base-uncased")
        encoder = CRISPEncoder(config)
        assert encoder.config.method == PruningMethod.C25

    @patch("crisp.models.encoder.AutoModel")
    def test_forward_shape(self, mock_automodel, default_config):
        """Test forward pass output shapes."""
        batch_size = 4
        seq_len = 128
        hidden_size = 768

        # Mock base model
        mock_model = Mock()
        mock_model.config.hidden_size = hidden_size
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
        mock_model.return_value = mock_output
        mock_automodel.from_pretrained.return_value = mock_model

        encoder = CRISPEncoder(default_config)

        # Create inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        output = encoder(input_ids, attention_mask, is_query=True)

        # Check shape based on method
        embeddings = output["embeddings"]
        assert embeddings.dim() == 3  # batch x num_tokens x hidden_dim
        assert embeddings.size(0) == batch_size
        assert embeddings.size(-1) == hidden_size

    @patch("crisp.models.encoder.AutoModel")
    def test_forward_with_padding(self, mock_automodel, default_config):
        """Test forward pass with padded sequences."""
        batch_size = 2
        seq_len = 50
        hidden_size = 768

        # Mock base model
        mock_model = Mock()
        mock_model.config.hidden_size = hidden_size
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
        mock_model.return_value = mock_output
        mock_automodel.from_pretrained.return_value = mock_model

        encoder = CRISPEncoder(default_config)

        # Create inputs with padding
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 30:] = 0  # Pad first sequence
        attention_mask[1, 40:] = 0  # Pad second sequence

        # Forward pass
        output = encoder(input_ids, attention_mask, is_query=False)

        # Should handle padding correctly
        embeddings = output["embeddings"]
        assert embeddings.shape[0] == batch_size
        assert torch.all(torch.isfinite(embeddings))

    @patch("crisp.models.encoder.AutoTokenizer")
    @patch("crisp.models.encoder.AutoModel")
    def test_gradient_checkpointing(self, mock_automodel, mock_autotokenizer):
        """Test gradient checkpointing configuration."""
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_automodel.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_autotokenizer.from_pretrained.return_value = mock_tokenizer

        # With gradient checkpointing
        config = CRISPConfig(gradient_checkpointing=True)
        encoder = CRISPEncoder(config)
        assert encoder is not None
        mock_model.gradient_checkpointing_enable.assert_called_once()

        # Without gradient checkpointing
        mock_model.reset_mock()
        config = CRISPConfig(gradient_checkpointing=False)
        CRISPEncoder(config)
        mock_model.gradient_checkpointing_enable.assert_not_called()

    @patch("crisp.models.encoder.AutoModel")
    def test_encode_method(self, mock_automodel, default_config):
        """Test the encode convenience method."""
        batch_size = 4
        hidden_size = 768

        # Mock base model
        mock_model = Mock()
        mock_model.config.hidden_size = hidden_size
        mock_automodel.from_pretrained.return_value = mock_model

        encoder = CRISPEncoder(default_config)

        # Mock forward method to return a dict
        expected_embeddings = torch.randn(batch_size, 8, hidden_size)
        expected_output = {
            "embeddings": expected_embeddings,
            "attention_mask": torch.ones(batch_size, 8),
        }
        encoder.forward = Mock(return_value=expected_output)

        # Mock tokenizer
        encoder.tokenizer = Mock()
        # Create actual tensors for tokenized output
        tokenized_data = {
            "input_ids": torch.randint(0, 1000, (batch_size, 50)),
            "attention_mask": torch.ones(batch_size, 50),
        }

        # Create a mock that acts like BatchEncoding (dict-like with .to() method)
        class MockBatchEncoding(dict):
            def to(self, device):
                return tokenized_data

        tokenized = MockBatchEncoding(tokenized_data)
        encoder.tokenizer.return_value = tokenized
        encoder.tokenizer.model_max_length = 512

        # Test encode
        texts = ["query 1", "query 2", "query 3", "query 4"]
        result = encoder.encode(texts, is_query=True)

        # Check that forward was called correctly
        encoder.forward.assert_called()
        assert "embeddings" in result
        assert "attention_mask" in result

    @patch("crisp.models.encoder.AutoModel")
    def test_device_handling(self, mock_automodel, default_config):
        """Test that model handles device placement correctly."""
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.return_value = mock_output
        mock_automodel.from_pretrained.return_value = mock_model

        encoder = CRISPEncoder(default_config)

        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)

        output = encoder(input_ids, attention_mask, is_query=True)

        # Should return tensor on same device as input
        embeddings = output["embeddings"]
        assert embeddings.device == input_ids.device

    def test_model_configuration_validation(self):
        """Test that model validates configuration properly."""
        # This should work
        config = CRISPConfig(
            model_name="bert-base-uncased", method=PruningMethod.C4X8, embedding_dim=768
        )

        # Model should be creatable with valid config
        # (Would need actual model for full test)
        assert config.model_name == "bert-base-uncased"
        assert config.method == PruningMethod.C4X8
