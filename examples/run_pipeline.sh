#!/bin/bash
# CRISP Pipeline: Train -> Infer -> Evaluate
# This script demonstrates the complete CRISP workflow

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=5e-5
METHOD="C4X8"
OUTPUT_DIR="output"
EVAL_ONLY=false
TRAIN_SAMPLES=500
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --train-samples)
            TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs NUM           Number of training epochs (default: 3)"
            echo "  --batch-size NUM       Batch size for training (default: 8)"
            echo "  --learning-rate RATE   Learning rate (default: 5e-5)"
            echo "  --method METHOD        Pruning method: C4X8, C8X32, TAIL_4X8, etc. (default: C4X8)"
            echo "  --output-dir DIR       Output directory (default: output)"
            echo "  --eval-only            Skip training, only run evaluation"
            echo "  --train-samples NUM    Number of training samples (default: 500)"
            echo "  --model-name NAME      Base model name (default: sentence-transformers/all-MiniLM-L6-v2)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CRISP Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Configuration:"
echo "  Method: $METHOD"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Training Samples: $TRAIN_SAMPLES"
echo "  Model: $MODEL_NAME"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Training
if [ "$EVAL_ONLY" = false ]; then
    echo -e "${GREEN}Step 1: Training CRISP Model${NC}"
    echo "=============================="
    
    # Create a training config file
    cat > "$OUTPUT_DIR/train_config.json" <<EOF
{
    "method": "$METHOD",
    "batch_size": $BATCH_SIZE,
    "learning_rate": $LEARNING_RATE,
    "epochs": $EPOCHS,
    "model_name": "$MODEL_NAME"
}
EOF
    
    # Run training
    uv run python train_crisp.py \
        --method "$METHOD" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --output-dir "$OUTPUT_DIR" \
        --num-samples "$TRAIN_SAMPLES" \
        --model-name "$MODEL_NAME"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed successfully${NC}"
    else
        echo -e "${RED}✗ Training failed${NC}"
        exit 1
    fi
else
    echo -e "${BLUE}Skipping training (--eval-only mode)${NC}"
fi

echo ""

# Step 2: Inference Demo
echo -e "${GREEN}Step 2: Running Inference Demo${NC}"
echo "==============================="

# Create inference script
cat > "$OUTPUT_DIR/run_inference.py" <<'EOF'
import sys
sys.path.append("crisp-py")

import torch
import json
from pathlib import Path
from crisp.config import CRISPConfig, PruningMethod
from crisp.models.lightning import CRISPModel
from crisp.losses.chamfer import ChamferSimilarity
from crisp.utils.device import get_device

def load_model(checkpoint_path, config_path):
    """Load trained CRISP model from checkpoint."""
    # Load config
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    
    # Create config
    config = CRISPConfig(
        method=PruningMethod[train_config['method']],
        model_name=train_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
    )
    
    # Create model
    model = CRISPModel(config)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, config

def run_inference(model, device):
    """Run inference on sample queries and documents."""
    # Sample data
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks",
    ]
    
    documents = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
        "A neural network is a series of algorithms that endeavors to recognize underlying relationships in data.",
        "Artificial intelligence is intelligence demonstrated by machines, in contrast to natural intelligence.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.",
    ]
    
    # Move model to device and eval mode
    model = model.to(device)
    model.eval()
    
    # Encode
    print("Encoding queries...")
    query_outputs = model.encode_queries(queries)
    
    print("Encoding documents...")
    doc_outputs = model.encode_documents(documents)
    
    # Print shapes
    print(f"\nEmbedding shapes:")
    print(f"  Queries: {query_outputs['embeddings'].shape}")
    print(f"  Documents: {doc_outputs['embeddings'].shape}")
    
    # Compute similarities
    chamfer_sim = ChamferSimilarity()
    similarities = chamfer_sim(
        query_outputs['embeddings'].to(device),
        query_outputs['attention_mask'].to(device),
        doc_outputs['embeddings'].to(device),
        doc_outputs['attention_mask'].to(device)
    )
    
    # Show results
    print("\nRetrieval Results:")
    print("-" * 50)
    for i, query in enumerate(queries):
        print(f"\nQuery: '{query}'")
        scores = similarities[i].cpu().numpy()
        top_indices = scores.argsort()[::-1][:3]
        
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank}. (score: {scores[idx]:.4f}) {documents[idx][:80]}...")

def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
    
    # Find latest checkpoint
    checkpoints = list(output_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        checkpoints = list(output_dir.glob("crisp_final.pt"))
    
    if not checkpoints:
        print("No checkpoint found!")
        return
    
    checkpoint_path = sorted(checkpoints)[-1]
    config_path = output_dir / "train_config.json"
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load model
    device = get_device()
    model, config = load_model(checkpoint_path, config_path)
    
    # Run inference
    run_inference(model, device)

if __name__ == "__main__":
    main()
EOF

# Run inference
uv run python "$OUTPUT_DIR/run_inference.py" "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Inference completed successfully${NC}"
else
    echo -e "${RED}✗ Inference failed${NC}"
fi

echo ""

# Step 3: Evaluation
echo -e "${GREEN}Step 3: Evaluation${NC}"
echo "==================="

# Create evaluation script
cat > "$OUTPUT_DIR/run_evaluation.py" <<'EOF'
import sys
sys.path.append("crisp-py")

import torch
import json
import numpy as np
from pathlib import Path
from crisp.config import CRISPConfig, PruningMethod
from crisp.models.lightning import CRISPModel
from crisp.evaluation.metrics import calculate_ndcg, calculate_map, calculate_recall

def load_model(checkpoint_path, config_path):
    """Load trained CRISP model from checkpoint."""
    # Load config
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    
    # Create config
    config = CRISPConfig(
        method=PruningMethod[train_config['method']],
        model_name=train_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
    )
    
    # Create model
    model = CRISPModel(config)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, config

def create_synthetic_eval_data():
    """Create synthetic evaluation data for demo."""
    # Create queries
    queries = {
        "q1": "What is machine learning?",
        "q2": "Explain deep learning",
        "q3": "How do neural networks work?",
        "q4": "What is artificial intelligence?",
        "q5": "Natural language processing applications",
    }
    
    # Create corpus
    corpus = {
        "d1": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "d2": "Deep learning is a machine learning technique using multi-layer neural networks.",
        "d3": "Neural networks are computing systems inspired by biological neural networks.",
        "d4": "Artificial intelligence is the simulation of human intelligence by machines.",
        "d5": "Natural language processing enables computers to understand human language.",
        "d6": "Supervised learning uses labeled data to train machine learning models.",
        "d7": "Unsupervised learning finds patterns in unlabeled data.",
        "d8": "Reinforcement learning trains agents through reward and punishment.",
        "d9": "Computer vision enables machines to interpret visual information.",
        "d10": "Speech recognition converts spoken language into text.",
    }
    
    # Create relevance judgments (qrels)
    qrels = {
        "q1": {"d1": 2, "d6": 1, "d7": 1},  # ML query
        "q2": {"d2": 2, "d3": 1},           # DL query
        "q3": {"d3": 2, "d2": 1},           # NN query
        "q4": {"d4": 2, "d1": 1},           # AI query
        "q5": {"d5": 2, "d10": 1},          # NLP query
    }
    
    return queries, corpus, qrels

def evaluate_model(model, device, queries, corpus, qrels):
    """Evaluate model on synthetic data."""
    import time
    from crisp.evaluation.beir import BEIREvaluator
    from crisp.losses.chamfer import ChamferSimilarity
    
    # Create evaluator
    evaluator = BEIREvaluator(
        encoder=model.encoder,
        config=model.config,
        device=device
    )
    
    # Run evaluation
    print("Running evaluation...")
    metrics = evaluator.evaluate(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        k_values=[1, 3, 5, 10],
        save_results=False
    )
    
    # Add speed evaluation
    print("\nMeasuring retrieval speed...")
    model.eval()
    with torch.no_grad():
        # Encode queries
        query_list = list(queries.values())
        start_time = time.time()
        query_outputs = model.encode_queries(query_list)
        query_encode_time = time.time() - start_time
        
        # Encode documents
        doc_list = list(corpus.values())
        start_time = time.time()
        doc_outputs = model.encode_documents(doc_list)
        doc_encode_time = time.time() - start_time
        
        # Compute similarities
        chamfer_sim = ChamferSimilarity()
        start_time = time.time()
        similarities = chamfer_sim(
            query_outputs['embeddings'].to(device),
            query_outputs['attention_mask'].to(device),
            doc_outputs['embeddings'].to(device),
            doc_outputs['attention_mask'].to(device)
        )
        similarity_time = time.time() - start_time
    
    # Add speed metrics to results
    metrics['speed'] = {
        'query_encoding_time': query_encode_time,
        'doc_encoding_time': doc_encode_time,
        'similarity_computation_time': similarity_time,
        'total_retrieval_time': query_encode_time + doc_encode_time + similarity_time,
        'queries_per_second': len(queries) / (query_encode_time + similarity_time),
        'docs_per_second': len(corpus) / (doc_encode_time + similarity_time)
    }
    
    return metrics

def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
    
    # Find latest checkpoint
    checkpoints = list(output_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        checkpoints = list(output_dir.glob("crisp_final.pt"))
    
    if not checkpoints:
        print("No checkpoint found!")
        return
    
    checkpoint_path = sorted(checkpoints)[-1]
    config_path = output_dir / "train_config.json"
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load model
    from crisp.utils.device import get_device
    device = get_device()
    model, config = load_model(checkpoint_path, config_path)
    
    # Create evaluation data
    queries, corpus, qrels = create_synthetic_eval_data()
    
    # Evaluate
    metrics = evaluate_model(model, device, queries, corpus, qrels)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    
    for metric, value in sorted(metrics.items()):
        if metric == 'speed':
            print(f"\nSpeed Metrics:")
            print(f"  Query encoding: {value['query_encoding_time']:.4f}s")
            print(f"  Document encoding: {value['doc_encoding_time']:.4f}s")
            print(f"  Similarity computation: {value['similarity_computation_time']:.4f}s")
            print(f"  Total retrieval time: {value['total_retrieval_time']:.4f}s")
            print(f"  Queries per second: {value['queries_per_second']:.2f}")
            print(f"  Documents per second: {value['docs_per_second']:.2f}")
        elif isinstance(value, dict) and 'mean' in value:
            print(f"{metric}: {value['mean']:.4f}")
        elif not isinstance(value, dict):
            print(f"{metric}: {value:.4f}")
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
EOF

# Run evaluation
uv run python "$OUTPUT_DIR/run_evaluation.py" "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Evaluation completed successfully${NC}"
else
    echo -e "${RED}✗ Evaluation failed${NC}"
fi

echo ""

# Step 4: Compare with Post-hoc Clustering
echo -e "${GREEN}Step 4: Compare CRISP vs Post-hoc Clustering${NC}"
echo "============================================"

# Run comparison demo
if [ -f "demo_crisp_vs_posthoc.py" ]; then
    echo "Running comparison demo..."
    uv run python demo_crisp_vs_posthoc.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Comparison completed successfully${NC}"
    else
        echo -e "${RED}✗ Comparison failed${NC}"
    fi
else
    echo -e "${BLUE}Comparison script not found, skipping...${NC}"
fi

echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pipeline Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Output files:"
echo "  - Model checkpoints: $OUTPUT_DIR/checkpoint_epoch_*.pt"
echo "  - Final model: $OUTPUT_DIR/crisp_final.pt"
echo "  - Training config: $OUTPUT_DIR/train_config.json"
echo "  - Evaluation results: $OUTPUT_DIR/evaluation_results.json"
echo ""
echo "To run inference with the trained model:"
echo "  uv run python $OUTPUT_DIR/run_inference.py $OUTPUT_DIR"
echo ""
echo "To run evaluation only:"
echo "  $0 --eval-only --output-dir $OUTPUT_DIR"