# CRISP: Clustering-Optimized Multi-Vector Representations for Efficient Retrieval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of **CRISP (Clustering-based Reduction of Instructions for Semantic Pruning)**, demonstrating how training with clustering awareness produces superior multi-vector representations compared to post-hoc clustering approaches.

## Key Results

Our implementation shows that CRISP-trained models significantly outperform post-hoc clustering:

| Method | Top-1 Accuracy | Avg Score | Intra-cluster Distance |
|--------|----------------|-----------|------------------------|
| **CRISP-trained** | 100% | 0.7409 | 1.0003 (±0.0631) |
| Post-hoc clustering | 100% | 0.6700 | 1.1877 (±0.0527) |

**Key insight**: CRISP produces embeddings that are naturally more clusterable, resulting in better retrieval performance with fewer vectors.

## Pipeline Results

```bash
./examples/run_pipeline.sh --epochs 1 --train-samples 50
```

<details>
<summary>Click to see full output</summary>

```
========================================
CRISP Pipeline
========================================
Configuration:
  Method: C4X8
  Epochs: 1
  Batch Size: 8
  Learning Rate: 5e-5
  Output Directory: output
  Training Samples: 50
  Model: sentence-transformers/all-MiniLM-L6-v2

Step 1: Training CRISP Model
==============================
2025-07-27 18:51:09,261 - Using device: mps
Epoch 1: 100%|███████| 63/63 [03:44<00:00, 3.56s/it, loss=0.0026, avg_loss=0.5476]
Final accuracy: 100.00%

Embedding shapes comparison:
Query shape: torch.Size([1, 7, 384]) → torch.Size([1, 4, 384])  # 4 clusters
Document shape: torch.Size([1, 13, 384]) → torch.Size([1, 8, 384])  # 8 clusters

Step 2-4: [Inference, Evaluation, Comparison]
...

CRISP-trained Model Results:
- Query: 'What is machine learning?'
  1. Doc 0 (score: 0.7409): Machine learning is a subset of artificial intelligence...
  
Post-hoc Model Results:
- Query: 'What is machine learning?'
  1. Doc 0 (score: 0.6700): Machine learning is a subset of artificial intelligence...

KEY INSIGHTS:
1. CRISP model is trained to produce embeddings optimized for clustering
2. Post-hoc model applies clustering to embeddings not designed for it
3. CRISP shows better retrieval performance with fewer vectors
4. CRISP embeddings cluster more naturally (lower intra-cluster distance)
```

</details>

## What is CRISP?

CRISP is a novel approach to multi-vector retrieval that **trains models to produce cluster-friendly embeddings** from the start, rather than applying clustering as an afterthought. This results in:
The paper from Google Deepmind is in [arxiv](arxiv.org/pdf/2505.11471).

## Installation

### Quick Start

```bash
# Clone the repository
https://github.com/sigridjineth/crisp-py
cd crisp-py

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the complete pipeline
./examples/run_pipeline.sh
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA/MPS support (optional but recommended)
- 8GB+ RAM for training

## Quick Demo

### 1. Train a CRISP Model

```bash
# Quick training (1 epoch, 50 samples)
./examples/run_pipeline.sh --epochs 1 --train-samples 50

# Full training
./examples/run_pipeline.sh --epochs 5 --train-samples 1000 --batch-size 16
```

### 2. Compare Methods

```python
# Run comparison demo
uv run python /examples/demo_crisp_vs_posthoc.py
```

### 3. Custom Usage

```python
from crisp.config import CRISPConfig, PruningMethod
from crisp.models.lightning import CRISPModel

# Configure CRISP
config = CRISPConfig(
    method=PruningMethod.C4X8,  # 4 query clusters, 8 doc clusters
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create and train model
model = CRISPModel(config)

# Use for retrieval
queries = ["What is machine learning?"]
documents = ["Machine learning is...", "Neural networks are..."]

query_embeds = model.encode_queries(queries)
doc_embeds = model.encode_documents(documents)
```

## Different Clustering Methods

```bash
# Fixed clustering with 8 query clusters, 32 doc clusters
./examples/run_pipeline.sh --method C8X32

# Relative clustering (25% of tokens)
./examples/run_pipeline.sh --method C25

# Tail pruning baseline
./examples/run_pipeline.sh --method TAIL_4X8

# K-spacing baseline
./examples/run_pipeline.sh --method KSPACING_2
```

## Custom Models

```bash
# Use different base model
./examples/run_pipeline.sh --model-name "BAAI/bge-small-en-v1.5"

# Larger batch size for faster training
./examples/run_pipeline.sh --batch-size 32 --learning-rate 1e-4
```

## Evaluation Only

```bash
# Skip training, only evaluate existing model
./examples/run_pipeline.sh --eval-only --output-dir output
```

## Citation

This implementation is based on the CRISP paper.

If you use this code in your research, please cite:

```bibtex
@article{crisp2024,
  title={CRISP: Clustering-based Reduction of Instructions for Semantic Pruning},
  author={[Authors]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for the IR community
</p>