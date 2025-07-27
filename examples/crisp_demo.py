"""
CRISP Demo: Comprehensive example showing CRISP vs non-CRISP comparison
and practical usage patterns.

This example demonstrates:
1. Comparing unpruned multi-vector vs various CRISP methods
2. How to use CRISP for encoding and retrieval
3. MPS/CUDA/CPU device support
"""

import torch
import numpy as np
import sys
import time
from typing import List, Dict, Tuple

sys.path.append("crisp-py")  # Add crisp-py to path

from crisp.config import CRISPConfig, PruningMethod
from crisp.models.lightning import CRISPModel
from crisp.losses.chamfer import ChamferSimilarity
from crisp.utils.device import get_device, log_device_info


def encode_and_retrieve(
    model: CRISPModel,
    queries: List[str],
    documents: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Encode queries and documents, then compute retrieval scores.
    
    Returns:
        - Similarity matrix [num_queries, num_docs]
        - Timing information
    """
    timings = {}
    
    # Encode queries
    start_time = time.time()
    with torch.no_grad():
        query_outputs = model.encode_queries(queries, batch_size=32)
        query_embeddings = query_outputs["embeddings"].to(device)
        query_mask = query_outputs["attention_mask"].to(device)
    timings["query_encoding"] = time.time() - start_time
    
    # Encode documents
    start_time = time.time()
    with torch.no_grad():
        doc_outputs = model.encode_documents(documents, batch_size=32)
        doc_embeddings = doc_outputs["embeddings"].to(device)
        doc_mask = doc_outputs["attention_mask"].to(device)
    timings["doc_encoding"] = time.time() - start_time
    
    # Compute similarities
    start_time = time.time()
    chamfer_sim = ChamferSimilarity()
    similarity_matrix = ChamferSimilarity.compute_retrieval_scores(
        query_embeddings, query_mask, doc_embeddings, doc_mask,
        batch_size=128
    )
    timings["similarity_computation"] = time.time() - start_time
    
    return similarity_matrix, timings


def print_retrieval_results(
    queries: List[str],
    documents: List[str],
    similarity_matrix: torch.Tensor,
    top_k: int = 3
):
    """Print top-k retrieval results for each query."""
    for i, query in enumerate(queries):
        print(f"\nQuery: '{query}'")
        scores = similarity_matrix[i].cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank}. (score: {scores[idx]:.4f}) {documents[idx][:80]}...")


def compare_crisp_methods():
    """Compare different CRISP methods with comprehensive examples."""
    
    # Get the best device (CUDA, MPS, or CPU)
    device = get_device()
    print("=" * 80)
    print("CRISP Demo: Comparing Multi-Vector Retrieval Methods")
    print("=" * 80)
    log_device_info(device)
    print()
    
    # Extended example data
    queries = [
        "What is machine learning?",
        "How does neural network work?",
        "What are the benefits of deep learning?",
        "Explain gradient descent optimization",
        "What is transfer learning?"
    ]
    
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "Neural networks are computing systems inspired by biological neural networks that constitute animal brains, consisting of interconnected nodes or neurons.",
        "Deep learning is a machine learning technique that teaches computers to learn by example through multiple layers of neural networks.",
        "Gradient descent is an optimization algorithm used to minimize the cost function in machine learning by iteratively moving in the direction of steepest descent.",
        "Transfer learning is a machine learning method where a model developed for one task is reused as the starting point for a model on a related task.",
        "Artificial intelligence encompasses machine learning, deep learning, and other techniques for building intelligent systems.",
        "Backpropagation is a method used in artificial neural networks to calculate the gradient of the loss function with respect to the weights.",
        "Convolutional neural networks (CNNs) are deep learning algorithms particularly effective for analyzing visual imagery.",
        "Recurrent neural networks (RNNs) are a class of neural networks well-suited for processing sequential data like text or time series.",
        "Supervised learning is a type of machine learning where the model is trained on labeled data with known outputs."
    ]
    
    # Define methods to compare
    methods = [
        (None, "Full Multi-Vector (No Pruning/Clustering)"),
        (PruningMethod.C4X8, "CRISP Fixed Clustering C4x8"),
        (PruningMethod.C8X32, "CRISP Fixed Clustering C8x32"),
        (PruningMethod.C25, "CRISP Relative Clustering 25%"),
        (PruningMethod.C50, "CRISP Relative Clustering 50%"),
        (PruningMethod.TAIL_4X8, "CRISP Tail Pruning 4x8"),
        (PruningMethod.TAIL_8X32, "CRISP Tail Pruning 8x32"),
        (PruningMethod.K2, "CRISP K-Spacing (k=2)"),
        (PruningMethod.K4, "CRISP K-Spacing (k=4)")
    ]
    
    results = []
    baseline_query_shape = None
    baseline_doc_shape = None
    
    for method, name in methods:
        print(f"\n{'=' * 80}")
        print(f"Method: {name}")
        print(f"{'=' * 80}")
        
        # Create config for this method
        if method is None:
            # No pruning/clustering - full multi-vector
            config = CRISPConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                temperature=0.05,
                normalize_embeddings=True
            )
            # Manually set method to None to avoid any pruning/clustering
            config.method = None
        else:
            config = CRISPConfig(method=method)
        
        # Initialize model
        print("Loading model...")
        model = CRISPModel(config)
        model = model.to(device)
        model.eval()
        
        # Encode and retrieve
        similarity_matrix, timings = encode_and_retrieve(
            model, queries, documents, device
        )
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"  Query encoding time: {timings['query_encoding']:.3f}s")
        print(f"  Document encoding time: {timings['doc_encoding']:.3f}s")
        print(f"  Similarity computation time: {timings['similarity_computation']:.3f}s")
        print(f"  Total time: {sum(timings.values()):.3f}s")
        
        # Get embedding shapes
        with torch.no_grad():
            sample_query = model.encode_queries([queries[0]], batch_size=1)
            sample_doc = model.encode_documents([documents[0]], batch_size=1)
        
        print(f"\nEmbedding shapes:")
        print(f"  Query embeddings: {sample_query['embeddings'].shape}")
        print(f"  Document embeddings: {sample_doc['embeddings'].shape}")
        
        # Store baseline shapes from first iteration (Full Multi-Vector)
        if method is None:
            baseline_query_shape = sample_query['embeddings'].shape[1]
            baseline_doc_shape = sample_doc['embeddings'].shape[1]
        
        # Calculate compression ratios
        if method is not None and baseline_query_shape is not None:
            query_compression = baseline_query_shape / sample_query['embeddings'].shape[1]
            doc_compression = baseline_doc_shape / sample_doc['embeddings'].shape[1]
            print(f"\nCompression ratios (vs actual baseline):")
            print(f"  Query compression: {query_compression:.1f}x")
            print(f"  Document compression: {doc_compression:.1f}x")
        
        # Show top retrieval results for first query
        print(f"\nTop-3 results for first query:")
        print_retrieval_results(queries[:1], documents, similarity_matrix[:1], top_k=3)
        
        # Store results for comparison
        results.append({
            'method': name,
            'similarity_matrix': similarity_matrix.cpu(),
            'timings': timings,
            'query_shape': sample_query['embeddings'].shape,
            'doc_shape': sample_doc['embeddings'].shape
        })
        
        # Memory cleanup
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            # MPS doesn't have empty_cache, but we can synchronize
            torch.mps.synchronize()
    
    # Summary comparison
    print(f"\n\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}")
    
    print("\nMethod Performance Summary:")
    print(f"{'Method':<40} {'Total Time (s)':<15} {'Query Shape':<20} {'Doc Shape':<20}")
    print("-" * 95)
    
    for result in results:
        total_time = sum(result['timings'].values())
        print(f"{result['method']:<40} {total_time:<15.3f} {str(result['query_shape']):<20} {str(result['doc_shape']):<20}")
    
    # Compare retrieval quality (using first query as example)
    print("\nRetrieval Quality Comparison (Query: '{}')".format(queries[0]))
    print(f"{'Method':<40} {'Top-1 Score':<15} {'Top-3 Avg Score':<15}")
    print("-" * 70)
    
    for result in results:
        scores = result['similarity_matrix'][0].numpy()
        top_scores = np.sort(scores)[::-1][:3]
        print(f"{result['method']:<40} {top_scores[0]:<15.4f} {np.mean(top_scores):<15.4f}")
    
    print(f"\n{'=' * 80}")
    print("Demo completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    compare_crisp_methods()