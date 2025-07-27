from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PruningMethod(Enum):
    # pruning methods from CRISP paper
    TAIL_4X8 = "tail_4x8"
    TAIL_8X32 = "tail_8x32"
    K2 = "k2"
    K4 = "k4"
    C4X8 = "c4x8"
    C8X32 = "c8x32"
    C25 = "c25"
    C50 = "c50"


@dataclass
class CRISPConfig:
    # model stuff
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 2048
    projection_dim: Optional[int] = None  # For dimension reduction
    max_query_length: int = 512
    max_doc_length: int = 512
    gradient_checkpointing: bool = True
    normalize_embeddings: bool = True

    # Training parameters (from paper)
    method: PruningMethod = PruningMethod.C8X32
    temperature: float = 0.05
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 128
    accumulate_grad_batches: int = 1
    warmup_steps: int = 1000
    warmup_ratio: float = 0.05  # Proportion of warmup steps
    max_steps: int = 20000
    lr_scheduler_type: str = "cosine"

    # Optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # K-means parameters
    kmeans_iterations: int = 20
    kmeans_max_iterations: int = 50  # With convergence check
    use_faiss_clustering: bool = True
    log_clustering_stats: bool = False

    # BEIR evaluation
    use_instruction_prefix: bool = True
    beir_task: Optional[str] = None

    # Memory optimization
    chunk_size: int = 16  # For chunked loss computation
    mixed_precision: bool = True

    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    num_hard_negatives: int = 2  # Number of hard negatives per query
    num_negatives: int = 8  # Total number of negatives per query

    # Distributed training
    use_distributed: bool = False

    # Logging
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3

    def __post_init__(self):
        # some basic checks
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.kmeans_iterations > self.kmeans_max_iterations:
            raise ValueError(
                f"kmeans_iterations ({self.kmeans_iterations}) must be <= "
                f"kmeans_max_iterations ({self.kmeans_max_iterations})"
            )

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        import json
        from pathlib import Path

        config_dict = {
            k: v.value if isinstance(v, Enum) else v for k, v in self.__dict__.items()
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
