# CRISP Pipeline Guide

This guide explains how to use the `run_pipeline.sh` script to run the complete CRISP workflow: training, inference, and evaluation.

## Quick Start

```bash
# Run the complete pipeline with default settings
./run_pipeline.sh

# Run with custom settings
./run_pipeline.sh --epochs 5 --batch-size 16 --method C8X32

# Skip training and only run evaluation on existing model
./run_pipeline.sh --eval-only --output-dir output
```

## Pipeline Steps

### 1. Training
- Trains a CRISP model from scratch
- Uses synthetic retrieval data for demonstration
- Saves checkpoints after each epoch

### 2. Inference
- Loads the trained model
- Runs inference on sample queries
- Shows retrieval results with scores

### 3. Evaluation
- Evaluates on synthetic test data
- Computes standard IR metrics (NDCG, MAP, Recall)
- Saves results to JSON

### 4. Comparison (Optional)
- Compares CRISP-trained model vs post-hoc clustering
- Shows performance differences

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs` | Number of training epochs | 3 |
| `--batch-size` | Batch size for training | 8 |
| `--learning-rate` | Learning rate | 5e-5 |
| `--method` | Pruning method (C4X8, C8X32, TAIL_4X8, etc.) | C4X8 |
| `--output-dir` | Output directory for models and results | output |
| `--eval-only` | Skip training, only run evaluation | false |
| `--train-samples` | Number of training samples | 500 |
| `--model-name` | Base model to use | sentence-transformers/all-MiniLM-L6-v2 |

## Output Files

After running the pipeline, you'll find:

- `output/checkpoint_epoch_*.pt` - Model checkpoints
- `output/crisp_final.pt` - Final trained model
- `output/train_config.json` - Training configuration
- `output/evaluation_results.json` - Evaluation metrics
- `output/run_inference.py` - Script to run inference
- `output/run_evaluation.py` - Script to run evaluation

## Examples

### Example 1: Quick Training and Evaluation
```bash
./run_pipeline.sh --epochs 2 --train-samples 100
```

### Example 2: Different Clustering Methods
```bash
# Fixed clustering with 8 query clusters, 32 doc clusters
./run_pipeline.sh --method C8X32

# Tail pruning
./run_pipeline.sh --method TAIL_4X8

# K-spacing
./run_pipeline.sh --method KSPACING_2
```

### Example 3: Evaluation Only
```bash
# Train first
./run_pipeline.sh --epochs 5 --output-dir models/experiment1

# Later, evaluate only
./run_pipeline.sh --eval-only --output-dir models/experiment1
```

### Example 4: Custom Model
```bash
# Use a different base model
./run_pipeline.sh --model-name "BAAI/bge-small-en-v1.5" --epochs 3
```

## Performance Tips

1. **GPU Usage**: The script automatically detects and uses GPU if available
2. **Batch Size**: Increase batch size for faster training if you have enough memory
3. **Learning Rate**: Lower learning rates often work better for larger models

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 4`
- Use smaller model: `--model-name sentence-transformers/all-MiniLM-L6-v2`

### Training Too Slow
- Reduce training samples: `--train-samples 100`
- Reduce epochs: `--epochs 1`

### No Checkpoint Found
- Check output directory path
- Ensure training completed successfully

## Advanced Usage

### Custom Evaluation Data
Modify `run_evaluation.py` in the output directory to use your own evaluation data instead of synthetic data.

### BEIR Evaluation
To evaluate on real BEIR datasets, modify the evaluation script to use the BEIR evaluator with downloaded datasets.

### Production Training
For production use:
1. Use more training data
2. Train for more epochs
3. Use validation set for early stopping
4. Implement learning rate scheduling