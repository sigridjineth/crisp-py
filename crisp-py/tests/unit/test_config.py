"""
Unit tests for CRISP configuration module.
"""

from dataclasses import fields

import pytest
from crisp.config import CRISPConfig, PruningMethod


class TestPruningMethod:
    """Test PruningMethod enum."""

    def test_all_methods_defined(self):
        """Test that all 8 pruning methods from the paper are defined."""
        expected_methods = {
            "TAIL_4X8",
            "TAIL_8X32",
            "K2",
            "K4",
            "C4X8",
            "C8X32",
            "C25",
            "C50",
        }
        actual_methods = {method.name for method in PruningMethod}
        assert actual_methods == expected_methods

    def test_method_values(self):
        """Test that method values follow naming convention."""
        assert PruningMethod.TAIL_4X8.value == "tail_4x8"
        assert PruningMethod.TAIL_8X32.value == "tail_8x32"
        assert PruningMethod.K2.value == "k2"
        assert PruningMethod.K4.value == "k4"
        assert PruningMethod.C4X8.value == "c4x8"
        assert PruningMethod.C8X32.value == "c8x32"
        assert PruningMethod.C25.value == "c25"
        assert PruningMethod.C50.value == "c50"

    def test_method_from_string(self):
        """Test creating method from string value."""
        assert PruningMethod("tail_4x8") == PruningMethod.TAIL_4X8
        assert PruningMethod("k2") == PruningMethod.K2
        assert PruningMethod("c25") == PruningMethod.C25


class TestCRISPConfig:
    """Test CRISPConfig dataclass."""

    def test_default_values(self):
        """Test that default values match paper specifications."""
        config = CRISPConfig()

        # Model defaults
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding_dim == 2048
        assert config.max_query_length == 512
        assert config.max_doc_length == 512

        # Training defaults
        assert config.method == PruningMethod.C8X32
        assert config.temperature == 0.05
        assert config.learning_rate == 2e-5
        assert config.weight_decay == 0.01
        assert config.batch_size == 128

        # K-means defaults
        assert config.kmeans_iterations == 20
        assert config.kmeans_max_iterations == 50

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = CRISPConfig(
            model_name="bert-base-uncased",
            embedding_dim=768,
            method=PruningMethod.TAIL_4X8,
            temperature=0.1,
            batch_size=32,
        )

        assert config.model_name == "bert-base-uncased"
        assert config.embedding_dim == 768
        assert config.method == PruningMethod.TAIL_4X8
        assert config.temperature == 0.1
        assert config.batch_size == 32

    def test_post_init_validation(self):
        """Test that post-init validation works correctly."""
        # Valid config should not raise
        config = CRISPConfig(kmeans_iterations=10, kmeans_max_iterations=20)
        assert config.kmeans_iterations == 10

        # Invalid kmeans iterations
        with pytest.raises(ValueError, match="kmeans_iterations.*must be <="):
            CRISPConfig(kmeans_iterations=30, kmeans_max_iterations=20)

        # Invalid temperature
        with pytest.raises(ValueError, match="temperature must be positive"):
            CRISPConfig(temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            CRISPConfig(temperature=-0.1)

        # Invalid chunk size
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            CRISPConfig(chunk_size=0)

    def test_all_fields_have_defaults(self):
        """Test that all fields have default values (no required fields)."""
        # Should be able to create config without any arguments
        config = CRISPConfig()

        # Check that all fields have values
        for field in fields(CRISPConfig):
            assert hasattr(config, field.name)

    def test_dataclass_features(self):
        """Test dataclass features like equality and repr."""
        config1 = CRISPConfig(batch_size=64)
        config2 = CRISPConfig(batch_size=64)
        config3 = CRISPConfig(batch_size=128)

        # Equality
        assert config1 == config2
        assert config1 != config3

        # Repr
        repr_str = repr(config1)
        assert "CRISPConfig" in repr_str
        assert "batch_size=64" in repr_str

    def test_memory_optimization_settings(self):
        """Test memory optimization configuration options."""
        config = CRISPConfig(
            gradient_checkpointing=True, mixed_precision=True, chunk_size=8
        )

        assert config.gradient_checkpointing is True
        assert config.mixed_precision is True
        assert config.chunk_size == 8

    def test_distributed_settings(self):
        """Test distributed training configuration."""
        config = CRISPConfig(use_distributed=True)
        assert config.use_distributed is True

        config = CRISPConfig(use_distributed=False)
        assert config.use_distributed is False

    def test_beir_settings(self):
        """Test BEIR evaluation settings."""
        config = CRISPConfig(use_instruction_prefix=True, beir_task="nfcorpus")

        assert config.use_instruction_prefix is True
        assert config.beir_task == "nfcorpus"

        # Test None default
        config_default = CRISPConfig()
        assert config_default.beir_task is None

    def test_scheduler_type_values(self):
        """Test scheduler type configuration."""
        config_cosine = CRISPConfig(lr_scheduler_type="cosine")
        assert config_cosine.lr_scheduler_type == "cosine"

        config_linear = CRISPConfig(lr_scheduler_type="linear")
        assert config_linear.lr_scheduler_type == "linear"

    @pytest.mark.parametrize(
        "field_name,invalid_value,error_match",
        [
            ("temperature", -1.0, "temperature must be positive"),
            ("temperature", 0.0, "temperature must be positive"),
            ("chunk_size", -1, "chunk_size must be positive"),
            ("chunk_size", 0, "chunk_size must be positive"),
            ("kmeans_iterations", 100, "kmeans_iterations.*must be <="),
        ],
    )
    def test_invalid_values(self, field_name, invalid_value, error_match):
        """Test various invalid configurations."""
        kwargs = {field_name: invalid_value}
        if field_name == "kmeans_iterations":
            kwargs["kmeans_max_iterations"] = 50

        with pytest.raises(ValueError, match=error_match):
            CRISPConfig(**kwargs)
