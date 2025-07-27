"""
CRISP Example Usage

This script demonstrates how to use the CRISP library for multi-vector
retrieval with clustering and pruning strategies.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crisp import (
    CRISPConfig,
    CRISPEncoder,
    CRISPLightningModule,
    FixedClustering,
    KMeansClustering,
    RelativeClustering,
    TailPruning,
    SpacingPruning,
    ChamferLoss,
    ContrastiveLoss
)
from crisp.data import CRISPDataset, BEIRDataset, get_crisp_collator
from crisp.evaluation import evaluate_on_beir
from crisp.utils import setup_logger, get_device, log_config


def train_crisp_model(args):
    """Train a CRISP model."""
    # Setup logging
    logger = setup_logger(
        name="crisp_training",
        log_file=args.output_dir / "training.log",
        use_json=True
    )
    
    # Configuration
    config = CRISPConfig(
        model_name_or_path=args.model_name,
        max_query_length=args.max_query_length,
        max_doc_length=args.max_doc_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        num_negatives=args.num_negatives,
        num_hard_negatives=args.num_hard_negatives,
        temperature=args.temperature,
        normalize_embeddings=True,
        use_fp16=args.fp16
    )
    
    log_config(config.to_dict(), logger)
    
    # Initialize encoder
    encoder = CRISPEncoder(config)
    logger.info(f"Initialized encoder from {args.model_name}")
    
    # Load training data
    train_dataset = CRISPDataset(
        data_path=args.data_path,
        tokenizer=encoder.tokenizer,
        config=config,
        mode="train"
    )
    
    # Create data loader
    collator = get_crisp_collator(
        pad_token_id=encoder.tokenizer.pad_token_id,
        max_query_length=config.max_query_length,
        max_doc_length=config.max_doc_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers
    )
    
    # Initialize clustering strategy if specified
    clustering_strategy = None
    if args.clustering_strategy == "fixed":
        clustering_strategy = FixedClustering(
            num_clusters_query=args.num_clusters_query,
            num_clusters_doc=args.num_clusters_doc
        )
    elif args.clustering_strategy == "kmeans":
        clustering_strategy = KMeansClustering(
            num_clusters_query=args.num_clusters_query,
            num_clusters_doc=args.num_clusters_doc
        )
    elif args.clustering_strategy == "relative":
        clustering_strategy = RelativeClustering(
            cluster_ratio=args.cluster_ratio
        )
        
    # Initialize pruning strategy if specified
    pruning_strategy = None
    if args.pruning_strategy == "tail":
        pruning_strategy = TailPruning(
            keep_tokens_query=args.keep_tokens_query,
            keep_tokens_doc=args.keep_tokens_doc
        )
    elif args.pruning_strategy == "spacing":
        pruning_strategy = SpacingPruning(
            spacing_factor=args.spacing_factor
        )
    
    # Initialize loss
    if args.loss_type == "chamfer":
        loss_fn = ChamferLoss(temperature=config.temperature)
    else:
        loss_fn = ContrastiveLoss(temperature=config.temperature)
    
    # Create Lightning module
    model = CRISPLightningModule(
        encoder=encoder,
        loss_fn=loss_fn,
        config=config,
        clustering_strategy=clustering_strategy,
        pruning_strategy=pruning_strategy
    )
    
    # Train using PyTorch Lightning
    import pytorch_lightning as pl
    
    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir / "checkpoints",
            filename="crisp-{epoch:02d}-{val_loss:.3f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step")
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16 if config.use_fp16 else 32,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        enable_checkpointing=True,
        logger=pl.loggers.TensorBoardLogger(args.output_dir / "logs")
    )
    
    # Train
    trainer.fit(model, train_loader)
    
    # Save final model
    model.encoder.save_pretrained(args.output_dir / "final_model")
    logger.info(f"Model saved to {args.output_dir / 'final_model'}")


def evaluate_crisp_model(args):
    """Evaluate a CRISP model on BEIR."""
    # Setup logging
    logger = setup_logger(
        name="crisp_evaluation",
        log_file=args.output_dir / "evaluation.log"
    )
    
    # Load model
    encoder = CRISPEncoder.from_pretrained(args.model_path)
    logger.info(f"Loaded model from {args.model_path}")
    
    # Initialize strategies
    clustering_strategy = None
    if args.clustering_strategy == "fixed":
        clustering_strategy = FixedClustering(
            num_clusters_query=args.num_clusters_query,
            num_clusters_doc=args.num_clusters_doc
        )
    elif args.clustering_strategy == "kmeans":
        clustering_strategy = KMeansClustering(
            num_clusters_query=args.num_clusters_query,
            num_clusters_doc=args.num_clusters_doc
        )
    elif args.clustering_strategy == "relative":
        clustering_strategy = RelativeClustering(
            cluster_ratio=args.cluster_ratio
        )
        
    pruning_strategy = None
    if args.pruning_strategy == "tail":
        pruning_strategy = TailPruning(
            keep_tokens_query=args.keep_tokens_query,
            keep_tokens_doc=args.keep_tokens_doc
        )
    elif args.pruning_strategy == "spacing":
        pruning_strategy = SpacingPruning(
            spacing_factor=args.spacing_factor
        )
    
    # Evaluate on BEIR
    results = evaluate_on_beir(
        model=encoder,
        dataset_names=args.datasets,
        data_path=args.beir_data_path,
        clustering_strategy=clustering_strategy,
        pruning_strategy=pruning_strategy,
        batch_size=args.batch_size,
        k_values=[1, 3, 5, 10, 100],
        save_results=True,
        results_dir=args.output_dir / "results"
    )
    
    # Print summary
    print("\nEvaluation Results:")
    print("=" * 80)
    
    for dataset, metrics in results.items():
        if "error" in metrics:
            print(f"{dataset}: ERROR - {metrics['error']}")
        else:
            ndcg10 = metrics.get("ndcg@10", {}).get("mean", 0)
            map_score = metrics.get("map", {}).get("mean", 0)
            print(f"{dataset}: NDCG@10={ndcg10:.4f}, MAP={map_score:.4f}")
            
    # Calculate average NDCG@10
    ndcg_scores = []
    for dataset, metrics in results.items():
        if "error" not in metrics and "ndcg@10" in metrics:
            ndcg_scores.append(metrics["ndcg@10"]["mean"])
            
    if ndcg_scores:
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
        print(f"\nAverage NDCG@10: {avg_ndcg:.4f}")


def encode_with_crisp(args):
    """Encode texts using a CRISP model."""
    # Load model
    encoder = CRISPEncoder.from_pretrained(args.model_path)
    
    # Load texts
    if args.input_file:
        with open(args.input_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = args.texts
        
    # Initialize strategies
    clustering_strategy = None
    if args.clustering_strategy:
        if args.clustering_strategy == "kmeans":
            clustering_strategy = KMeansClustering(
                num_clusters_query=args.num_clusters_query,
                num_clusters_doc=args.num_clusters_doc
            )
            
    # Encode texts
    device = get_device()
    encoder = encoder.to(device)
    
    print(f"Encoding {len(texts)} texts...")
    
    with torch.no_grad():
        embeddings = encoder.encode(
            texts,
            batch_size=args.batch_size,
            device=device,
            show_progress=True
        )
        
    # Apply clustering if specified
    if clustering_strategy:
        print("Applying clustering...")
        # Determine if texts are queries or documents based on length
        avg_length = sum(len(t.split()) for t in texts) / len(texts)
        is_query = avg_length < 20  # Simple heuristic
        
        clustered = clustering_strategy.cluster(
            embeddings["embeddings"],
            embeddings["attention_mask"],
            is_query=is_query
        )
        embeddings = clustered
        
    # Save embeddings
    output_path = args.output_dir / "embeddings.pt"
    torch.save({
        "embeddings": embeddings["embeddings"].cpu(),
        "attention_mask": embeddings["attention_mask"].cpu(),
        "texts": texts
    }, output_path)
    
    print(f"Embeddings saved to {output_path}")
    print(f"Shape: {embeddings['embeddings'].shape}")


def main():
    parser = argparse.ArgumentParser(description="CRISP Example Script")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train a CRISP model")
    train_parser.add_argument("--data-path", type=Path, required=True, help="Path to training data")
    train_parser.add_argument("--model-name", default="google/gemma-2b", help="Base model name")
    train_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    train_parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    train_parser.add_argument("--max-query-length", type=int, default=32, help="Max query length")
    train_parser.add_argument("--max-doc-length", type=int, default=512, help="Max document length")
    train_parser.add_argument("--num-negatives", type=int, default=7, help="Number of negatives")
    train_parser.add_argument("--num-hard-negatives", type=int, default=2, help="Number of hard negatives")
    train_parser.add_argument("--temperature", type=float, default=0.01, help="Temperature for loss")
    train_parser.add_argument("--loss-type", choices=["chamfer", "contrastive"], default="chamfer")
    train_parser.add_argument("--clustering-strategy", choices=["fixed", "kmeans", "relative"], help="Clustering strategy")
    train_parser.add_argument("--num-clusters-query", type=int, default=4, help="Number of query clusters")
    train_parser.add_argument("--num-clusters-doc", type=int, default=8, help="Number of document clusters")
    train_parser.add_argument("--cluster-ratio", type=float, default=0.25, help="Cluster ratio for relative clustering")
    train_parser.add_argument("--pruning-strategy", choices=["tail", "spacing"], help="Pruning strategy")
    train_parser.add_argument("--keep-tokens-query", type=int, default=4, help="Tokens to keep for queries")
    train_parser.add_argument("--keep-tokens-doc", type=int, default=8, help="Tokens to keep for documents")
    train_parser.add_argument("--spacing-factor", type=int, default=2, help="Spacing factor for pruning")
    train_parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    train_parser.add_argument("--num-workers", type=int, default=4, help="Number of data workers")
    
    # Evaluation arguments
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a CRISP model")
    eval_parser.add_argument("--model-path", type=Path, required=True, help="Path to model")
    eval_parser.add_argument("--beir-data-path", type=Path, required=True, help="Path to BEIR data")
    eval_parser.add_argument("--datasets", nargs="+", default=["arguana", "fiqa", "nfcorpus", "scifact"], help="BEIR datasets")
    eval_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    eval_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    eval_parser.add_argument("--clustering-strategy", choices=["fixed", "kmeans", "relative"], help="Clustering strategy")
    eval_parser.add_argument("--num-clusters-query", type=int, default=4, help="Number of query clusters")
    eval_parser.add_argument("--num-clusters-doc", type=int, default=8, help="Number of document clusters")
    eval_parser.add_argument("--cluster-ratio", type=float, default=0.25, help="Cluster ratio")
    eval_parser.add_argument("--pruning-strategy", choices=["tail", "spacing"], help="Pruning strategy")
    eval_parser.add_argument("--keep-tokens-query", type=int, default=4, help="Tokens to keep for queries")
    eval_parser.add_argument("--keep-tokens-doc", type=int, default=8, help="Tokens to keep for documents")
    eval_parser.add_argument("--spacing-factor", type=int, default=2, help="Spacing factor")
    
    # Encoding arguments
    encode_parser = subparsers.add_parser("encode", help="Encode texts")
    encode_parser.add_argument("--model-path", type=Path, required=True, help="Path to model")
    encode_parser.add_argument("--texts", nargs="+", help="Texts to encode")
    encode_parser.add_argument("--input-file", type=Path, help="File with texts to encode")
    encode_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    encode_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    encode_parser.add_argument("--clustering-strategy", choices=["kmeans"], help="Apply clustering")
    encode_parser.add_argument("--num-clusters-query", type=int, default=4, help="Number of query clusters")
    encode_parser.add_argument("--num-clusters-doc", type=int, default=8, help="Number of document clusters")
    
    args = parser.parse_args()
    
    # Create output directory
    if hasattr(args, 'output_dir'):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run command
    if args.command == "train":
        train_crisp_model(args)
    elif args.command == "evaluate":
        evaluate_crisp_model(args)
    elif args.command == "encode":
        encode_with_crisp(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()