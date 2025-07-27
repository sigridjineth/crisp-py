#!/usr/bin/env python3
"""
Train a CRISP model with clustering during training.

This script demonstrates the key difference between CRISP and post-hoc clustering:
- CRISP trains the model to produce embeddings optimized for clustering
- Post-hoc clustering applies clustering after training on unprepared embeddings
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add crisp-py to path
import sys
sys.path.append("crisp-py")

from crisp.config import CRISPConfig, PruningMethod
from crisp.models.lightning import CRISPModel
from crisp.losses.chamfer import ChamferSimilarity
from crisp.losses.contrastive import InfoNCELoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleRetrievalDataset(Dataset):
    """Simple dataset for demonstration with query-document pairs."""
    
    def __init__(self, num_samples: int = 1000, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Create synthetic query-document pairs
        self.queries = []
        self.documents = []
        self.negatives = []
        
        # Topics for synthetic data
        topics = [
            "machine learning", "neural networks", "deep learning", 
            "artificial intelligence", "computer vision", "natural language processing",
            "reinforcement learning", "supervised learning", "unsupervised learning",
            "data science", "statistics", "algorithms", "optimization",
            "gradient descent", "backpropagation", "transformers", "attention mechanism",
            "convolutional networks", "recurrent networks", "generative models"
        ]
        
        templates = [
            "What is {}?",
            "How does {} work?",
            "Explain {} in simple terms",
            "What are the applications of {}?",
            "Compare {} with other methods",
            "What are the benefits of {}?",
            "When should I use {}?",
            "What are the limitations of {}?",
            "How to implement {}?",
            "Best practices for {}"
        ]
        
        doc_templates = [
            "{} is a technique in artificial intelligence that...",
            "{} works by processing data through multiple layers...",
            "The main advantage of {} is its ability to...",
            "{} has been successfully applied in areas such as...",
            "When implementing {}, it's important to consider...",
            "{} differs from traditional methods by...",
            "Recent advances in {} have led to improvements in...",
            "{} is particularly useful for solving problems involving...",
            "The key components of {} include...",
            "{} has revolutionized the field by enabling..."
        ]
        
        for i in range(num_samples):
            # Pick a topic
            topic = random.choice(topics)
            
            # Create query
            query_template = random.choice(templates)
            query = query_template.format(topic)
            self.queries.append(query)
            
            # Create positive document
            doc_template = random.choice(doc_templates)
            doc = doc_template.format(topic)
            # Add some extra content
            doc += f" This technology has shown remarkable results in various domains. " \
                   f"Researchers have found that {topic} can significantly improve performance. " \
                   f"The future of {topic} looks promising with ongoing developments."
            self.documents.append(doc)
            
            # Create negative documents (from different topics)
            neg_docs = []
            other_topics = [t for t in topics if t != topic]
            for _ in range(4):  # 4 negatives per query
                neg_topic = random.choice(other_topics)
                neg_template = random.choice(doc_templates)
                neg_doc = neg_template.format(neg_topic)
                neg_doc += f" This approach is different from {topic} in several ways. " \
                           f"Unlike {topic}, this method focuses on different aspects."
                neg_docs.append(neg_doc)
            self.negatives.append(neg_docs)
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return {
            'query': self.queries[idx],
            'positive': self.documents[idx],
            'negatives': self.negatives[idx]
        }


def create_training_batch(batch: Dict[str, List], device: torch.device) -> Dict[str, List[str]]:
    """Prepare batch for training."""
    # DataLoader with collate_fn returns a dict of lists
    queries = batch['query'] if isinstance(batch, dict) else [item['query'] for item in batch]
    positives = batch['positive'] if isinstance(batch, dict) else [item['positive'] for item in batch]
    negatives = []
    
    if isinstance(batch, dict):
        for neg_list in batch['negatives']:
            negatives.extend(neg_list)
    else:
        for item in batch:
            negatives.extend(item['negatives'])
    
    # In-batch negatives: use other positives as negatives
    all_docs = positives + negatives
    
    return {
        'queries': queries,
        'documents': all_docs,
        'labels': torch.arange(len(queries), device=device)  # Positive is at index i for query i
    }


def train_epoch(
    model: CRISPModel, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # Prepare batch
        batch_data = create_training_batch(batch, device)
        
        # Encode queries and documents
        # Don't use encode_queries/encode_documents in training as they put model in eval mode
        # Instead, manually encode with training mode preserved
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update stats
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/num_batches:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / num_batches


def evaluate_model(model: CRISPModel, test_queries: List[str], test_docs: List[str], device: torch.device):
    """Evaluate model performance."""
    model.eval()
    
    with torch.no_grad():
        # Encode
        query_outputs = model.encode_queries(test_queries)
        doc_outputs = model.encode_documents(test_docs)
        
        # Compute similarities
        query_embeds = query_outputs['embeddings'].to(device)
        query_mask = query_outputs['attention_mask'].to(device)
        doc_embeds = doc_outputs['embeddings'].to(device)
        doc_mask = doc_outputs['attention_mask'].to(device)
        
        chamfer_sim = ChamferSimilarity()
        similarities = chamfer_sim(query_embeds, query_mask, doc_embeds, doc_mask)
        
        # Get top-1 accuracy (assuming first doc is correct for first query, etc.)
        predictions = similarities.argmax(dim=1)
        correct = (predictions == torch.arange(len(test_queries), device=device)).sum().item()
        accuracy = correct / len(test_queries)
        
        return accuracy, similarities


def main():
    parser = argparse.ArgumentParser(description="Train CRISP model")
    parser.add_argument("--method", type=str, default="C8X32", 
                       help="CRISP method (C4X8, C8X32, etc.)")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, 
                       help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="crisp_models", 
                       help="Output directory")
    parser.add_argument("--num-samples", type=int, default=500,
                       help="Number of training samples")
    parser.add_argument("--model-name", type=str, 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Base model name")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    logger.info("Creating synthetic training dataset...")
    train_dataset = SimpleRetrievalDataset(num_samples=args.num_samples)
    
    # Custom collate function
    def collate_fn(batch):
        return {
            'query': [item['query'] for item in batch],
            'positive': [item['positive'] for item in batch],
            'negatives': [item['negatives'] for item in batch]
        }
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Create test set
    test_queries = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain deep learning",
        "What is gradient descent?",
        "How do transformers work?"
    ]
    test_docs = [
        "Machine learning is a subset of artificial intelligence...",
        "Neural networks are computing systems inspired by biological neural networks...",
        "Deep learning is a machine learning technique using multiple layers...",
        "Gradient descent is an optimization algorithm for finding minima...",
        "Transformers are a neural architecture based on self-attention..."
    ]
    
    # Create config
    method_map = {
        "C4X8": PruningMethod.C4X8,
        "C8X32": PruningMethod.C8X32,
        "C25": PruningMethod.C25,
        "C50": PruningMethod.C50
    }
    
    config = CRISPConfig(
        method=method_map.get(args.method, PruningMethod.C8X32),
        model_name=args.model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        warmup_steps=100
    )
    
    # Create model
    logger.info(f"Creating CRISP model with method: {args.method}")
    model = CRISPModel(config)
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    training_stats = []
    
    # Initial evaluation
    accuracy, _ = evaluate_model(model, test_queries, test_docs, device)
    logger.info(f"Initial accuracy: {accuracy:.2%}")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # Evaluate
        accuracy, similarities = evaluate_model(model, test_queries, test_docs, device)
        
        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")
        
        training_stats.append({
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy
        })
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'loss': avg_loss,
            'accuracy': accuracy
        }, checkpoint_path)
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Model saved to {final_path}")
    
    # Save training stats
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    # Final evaluation with detailed results
    logger.info("\nFinal evaluation on test set:")
    accuracy, similarities = evaluate_model(model, test_queries, test_docs, device)
    logger.info(f"Final accuracy: {accuracy:.2%}")
    
    # Show similarity scores
    print("\nSimilarity scores (query -> document):")
    for i, query in enumerate(test_queries):
        scores = similarities[i].cpu().numpy()
        top_idx = scores.argmax()
        print(f"\nQuery: {query}")
        print(f"Top match: Doc {top_idx} (score: {scores[top_idx]:.4f})")
        print(f"Correct doc score: {scores[i]:.4f}")
    
    # Compare shapes with and without CRISP
    logger.info("\nEmbedding shapes comparison:")
    with torch.no_grad():
        sample_query_output = model.encode_queries([test_queries[0]])
        sample_doc_output = model.encode_documents([test_docs[0]])
        
        print(f"Query shape: {sample_query_output['embeddings'].shape}")
        print(f"Document shape: {sample_doc_output['embeddings'].shape}")
    
    logger.info(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()