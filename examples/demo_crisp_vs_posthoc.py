#!/usr/bin/env python3
"""
Demonstration: CRISP-trained model vs Post-hoc clustering

This script shows the key difference between:
1. CRISP: Model trained with clustering during training (learns clusterable embeddings)
2. Post-hoc: Pre-trained model with clustering applied after (not optimized for clustering)
"""

import sys
sys.path.append("crisp-py")

import torch
import numpy as np
import time
from pathlib import Path

from crisp.config import CRISPConfig, PruningMethod
from crisp.models.lightning import CRISPModel
from crisp.losses.chamfer import ChamferSimilarity
from crisp.utils.device import get_device, log_device_info


def train_crisp_model(device: torch.device, epochs: int = 2):
    """Train a small CRISP model for demonstration."""
    print("\n" + "="*80)
    print("TRAINING CRISP MODEL (with clustering during training)")
    print("="*80)
    
    # Import training utilities
    from train_crisp import SimpleRetrievalDataset, create_training_batch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR
    from crisp.losses.contrastive import InfoNCELoss
    from crisp.losses.chamfer import ChamferSimilarity
    from tqdm import tqdm
    
    # Create config for CRISP training
    config = CRISPConfig(
        method=PruningMethod.C4X8,  # Use 4 clusters for queries, 8 for documents
        learning_rate=5e-5,
        batch_size=8,
        warmup_steps=50
    )
    
    # Create model
    model = CRISPModel(config)
    model = model.to(device)
    
    # Create small training dataset
    train_dataset = SimpleRetrievalDataset(num_samples=100)  # Small dataset for quick training
    
    # Custom collate function
    def collate_fn(batch):
        return {
            'query': [item['query'] for item in batch],
            'positive': [item['positive'] for item in batch],
            'negatives': [item['negatives'] for item in batch]
        }
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=50)
    
    # Train for a few epochs
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Prepare batch
            optimizer.zero_grad()
            batch_data = create_training_batch(batch, device)
            
            # Don't use encode_queries/encode_documents in training as they put model in eval mode
            queries_text = batch_data['queries']
            docs_text = batch_data['documents']
            
            # Tokenize
            query_tokens = model.encoder.tokenizer(
                queries_text,
                padding=True,
                truncation=True,
                max_length=model.config.max_query_length,
                return_tensors="pt"
            ).to(device)
            
            doc_tokens = model.encoder.tokenizer(
                docs_text,
                padding=True,
                truncation=True,
                max_length=model.config.max_doc_length,
                return_tensors="pt"
            ).to(device)
            
            # Forward through model (preserves training mode)
            query_outputs = model.forward(
                query_tokens['input_ids'],
                query_tokens['attention_mask'],
                is_query=True
            )
            
            doc_outputs = model.forward(
                doc_tokens['input_ids'],
                doc_tokens['attention_mask'],
                is_query=False
            )
            
            # Move to device
            query_embeds = query_outputs['embeddings'].to(device)
            query_mask = query_outputs['attention_mask'].to(device)
            doc_embeds = doc_outputs['embeddings'].to(device)
            doc_mask = doc_outputs['attention_mask'].to(device)
            
            # Compute similarities using Chamfer
            chamfer_sim = ChamferSimilarity()
            similarities = chamfer_sim(query_embeds, query_mask, doc_embeds, doc_mask)
            
            # Compute loss
            loss_fn = InfoNCELoss(temperature=0.05)
            loss = loss_fn(similarities, batch_data['labels'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update stats
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        print(f"Epoch {epoch}: Average Loss = {total_loss/num_batches:.4f}")
    
    return model


def create_posthoc_model(device: torch.device):
    """Create a model for post-hoc clustering (no CRISP training)."""
    print("\n" + "="*80)
    print("CREATING POST-HOC MODEL (no clustering during training)")
    print("="*80)
    
    # Create config with same clustering parameters but no training
    config = CRISPConfig(
        method=PruningMethod.C4X8,  # Same clustering as CRISP model
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Pre-trained model
    )
    
    # Create model (no training, just load pre-trained)
    model = CRISPModel(config)
    model = model.to(device)
    model.eval()
    
    print("Loaded pre-trained model for post-hoc clustering")
    
    return model


def compare_models(crisp_model: CRISPModel, posthoc_model: CRISPModel, device: torch.device):
    """Compare CRISP-trained model with post-hoc clustering model."""
    print("\n" + "="*80)
    print("COMPARING CRISP vs POST-HOC CLUSTERING")
    print("="*80)
    
    # Test data
    test_queries = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain deep learning in simple terms",
        "What is gradient descent optimization?",
        "How do transformers work in NLP?"
    ]
    
    test_documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "Neural networks are computing systems inspired by biological neural networks that constitute animal brains, consisting of interconnected nodes or neurons.",
        "Deep learning is a machine learning technique that teaches computers to learn by example through multiple layers of neural networks.",
        "Gradient descent is an optimization algorithm used to minimize the cost function in machine learning by iteratively moving in the direction of steepest descent.",
        "Transformers are a neural architecture that relies on self-attention mechanisms to process sequential data, revolutionizing natural language processing.",
        "Artificial intelligence encompasses machine learning, deep learning, and other techniques for building intelligent systems.",
        "Backpropagation is a method used in artificial neural networks to calculate the gradient of the loss function with respect to the weights.",
        "Convolutional neural networks (CNNs) are deep learning algorithms particularly effective for analyzing visual imagery.",
        "Recurrent neural networks (RNNs) are a class of neural networks well-suited for processing sequential data like text or time series.",
        "Supervised learning is a type of machine learning where the model is trained on labeled data with known outputs."
    ]
    
    # Similarity computation
    chamfer_sim = ChamferSimilarity()
    
    # Store timing results
    timing_results = {}
    
    for model_name, model in [("CRISP-trained", crisp_model), ("Post-hoc", posthoc_model)]:
        print(f"\n{'-'*40}")
        print(f"{model_name} Model Results:")
        print(f"{'-'*40}")
        
        model.eval()
        with torch.no_grad():
            # Time encoding
            start_time = time.time()
            query_outputs = model.encode_queries(test_queries)
            query_encode_time = time.time() - start_time
            
            start_time = time.time()
            doc_outputs = model.encode_documents(test_documents)
            doc_encode_time = time.time() - start_time
            
            # Get shapes
            print(f"\nEmbedding shapes:")
            print(f"  Queries: {query_outputs['embeddings'].shape} (clustered from ~7-10 tokens)")
            print(f"  Documents: {doc_outputs['embeddings'].shape} (clustered from ~20-30 tokens)")
            
            # Move to device
            query_embeds = query_outputs['embeddings'].to(device)
            query_mask = query_outputs['attention_mask'].to(device)
            doc_embeds = doc_outputs['embeddings'].to(device)
            doc_mask = doc_outputs['attention_mask'].to(device)
            
            # Time similarity computation
            start_time = time.time()
            similarities = chamfer_sim(query_embeds, query_mask, doc_embeds, doc_mask)
            similarity_time = time.time() - start_time
            
            # Store timing results
            timing_results[model_name] = {
                'query_encode_time': query_encode_time,
                'doc_encode_time': doc_encode_time,
                'similarity_time': similarity_time,
                'total_time': query_encode_time + doc_encode_time + similarity_time
            }
            
            # Show top results for each query
            print(f"\nRetrieval Results (Top-3 for each query):")
            for i, query in enumerate(test_queries):
                scores = similarities[i].cpu().numpy()
                top_indices = np.argsort(scores)[::-1][:3]
                
                print(f"\nQuery: '{query}'")
                for rank, idx in enumerate(top_indices, 1):
                    print(f"  {rank}. Doc {idx} (score: {scores[idx]:.4f}): {test_documents[idx][:60]}...")
                
                # Check if correct document (index i) is in top-3
                correct_rank = np.where(np.argsort(scores)[::-1] == i)[0][0] + 1
                print(f"  [Correct document rank: {correct_rank}]")
            
            # Calculate metrics
            # Top-1 accuracy (assuming document i is correct for query i)
            predictions = similarities.argmax(dim=1).cpu().numpy()
            correct = sum(predictions[:5] == np.arange(5))
            accuracy = correct / 5
            
            # Mean Reciprocal Rank (MRR)
            mrr = 0
            for i in range(5):
                scores = similarities[i].cpu().numpy()
                rank = np.where(np.argsort(scores)[::-1] == i)[0][0] + 1
                mrr += 1.0 / rank
            mrr /= 5
            
            print(f"\nMetrics:")
            print(f"  Top-1 Accuracy: {accuracy:.2%}")
            print(f"  Mean Reciprocal Rank: {mrr:.4f}")
            
            # Analyze clustering behavior
            print(f"\nClustering Analysis:")
            # Look at variance within clusters (lower = better clustering)
            query_embed_numpy = query_embeds[0].cpu().numpy()  # First query
            distances = np.linalg.norm(query_embed_numpy[1:] - query_embed_numpy[0], axis=1)
            print(f"  Average intra-cluster distance (query): {np.mean(distances):.4f}")
            print(f"  Std dev of distances (query): {np.std(distances):.4f}")
            
            # Print timing results
            print(f"\nSpeed Metrics:")
            print(f"  Query encoding time: {timing_results[model_name]['query_encode_time']:.4f}s")
            print(f"  Document encoding time: {timing_results[model_name]['doc_encode_time']:.4f}s")
            print(f"  Similarity computation time: {timing_results[model_name]['similarity_time']:.4f}s")
            print(f"  Total retrieval time: {timing_results[model_name]['total_time']:.4f}s")
    
    # Print speed comparison summary
    if len(timing_results) == 2:
        print(f"\n{'='*80}")
        print("SPEED COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        crisp_times = timing_results['CRISP-trained']
        posthoc_times = timing_results['Post-hoc']
        
        # Calculate speedup
        query_speedup = posthoc_times['query_encode_time'] / crisp_times['query_encode_time']
        doc_speedup = posthoc_times['doc_encode_time'] / crisp_times['doc_encode_time']
        sim_speedup = posthoc_times['similarity_time'] / crisp_times['similarity_time']
        total_speedup = posthoc_times['total_time'] / crisp_times['total_time']
        
        print(f"\nTiming Comparison:")
        print(f"{'Task':<25} {'CRISP':>10} {'Post-hoc':>10} {'Speedup':>10}")
        print(f"{'-'*55}")
        print(f"{'Query Encoding':<25} {crisp_times['query_encode_time']:>10.4f}s {posthoc_times['query_encode_time']:>10.4f}s {query_speedup:>10.2f}x")
        print(f"{'Document Encoding':<25} {crisp_times['doc_encode_time']:>10.4f}s {posthoc_times['doc_encode_time']:>10.4f}s {doc_speedup:>10.2f}x")
        print(f"{'Similarity Computation':<25} {crisp_times['similarity_time']:>10.4f}s {posthoc_times['similarity_time']:>10.4f}s {sim_speedup:>10.2f}x")
        print(f"{'-'*55}")
        print(f"{'Total Retrieval Time':<25} {crisp_times['total_time']:>10.4f}s {posthoc_times['total_time']:>10.4f}s {total_speedup:>10.2f}x")
        
        print(f"\nKey Insights:")
        if total_speedup > 1:
            print(f"  - CRISP is {total_speedup:.2f}x faster overall than post-hoc clustering")
        else:
            print(f"  - Post-hoc is {1/total_speedup:.2f}x faster overall than CRISP")
        
        print(f"  - The speed difference is mainly in the {'encoding' if (query_speedup + doc_speedup) > sim_speedup else 'similarity computation'} phase")


def main():
    # Setup
    device = get_device()
    print("="*80)
    print("CRISP vs Post-hoc Clustering Demonstration")
    print("="*80)
    log_device_info(device)
    
    # Train CRISP model
    crisp_model = train_crisp_model(device, epochs=3)
    
    # Create post-hoc model
    posthoc_model = create_posthoc_model(device)
    
    # Compare
    compare_models(crisp_model, posthoc_model, device)
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. CRISP model is trained to produce embeddings optimized for clustering")
    print("2. Post-hoc model applies clustering to embeddings not designed for it")
    print("3. CRISP should show better retrieval performance with fewer vectors")
    print("4. CRISP embeddings should cluster more naturally (lower intra-cluster distance)")
    print("="*80)


if __name__ == "__main__":
    main()