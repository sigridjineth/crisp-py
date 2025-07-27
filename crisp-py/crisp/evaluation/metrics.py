"""Evaluation metrics for information retrieval."""

import math
from typing import Dict, List, Union

import numpy as np
import torch


def calculate_ndcg(
    retrieved: Dict[str, float], relevant: Dict[str, int], k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).

    Args:
        retrieved: Dictionary mapping doc IDs to scores
        relevant: Dictionary mapping doc IDs to relevance labels
        k: Cutoff for evaluation

    Returns:
        NDCG@k score
    """
    # Sort retrieved documents by score
    sorted_docs = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)[:k]

    # Calculate DCG
    dcg = 0.0
    for i, (doc_id, _) in enumerate(sorted_docs):
        if doc_id in relevant:
            rel = relevant[doc_id]
            dcg += (2**rel - 1) / math.log2(i + 2)

    # Calculate ideal DCG
    ideal_rels = sorted([rel for rel in relevant.values()], reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    # Return NDCG
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def calculate_map(retrieved: Dict[str, float], relevant: Dict[str, int]) -> float:
    """
    Calculate Mean Average Precision (MAP).

    Args:
        retrieved: Dictionary mapping doc IDs to scores
        relevant: Dictionary mapping doc IDs to relevance labels

    Returns:
        MAP score
    """
    # Sort retrieved documents by score
    sorted_docs = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)

    # Calculate average precision
    num_relevant = 0
    sum_precision = 0.0

    for i, (doc_id, _) in enumerate(sorted_docs):
        if doc_id in relevant and relevant[doc_id] > 0:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            sum_precision += precision_at_i

    # Return MAP
    if len(relevant) == 0:
        return 0.0
    return sum_precision / len(relevant)


def calculate_recall(
    retrieved: Dict[str, float], relevant: Dict[str, int], k: int
) -> float:
    """
    Calculate Recall@k.

    Args:
        retrieved: Dictionary mapping doc IDs to scores
        relevant: Dictionary mapping doc IDs to relevance labels
        k: Cutoff for evaluation

    Returns:
        Recall@k score
    """
    # Get top-k retrieved documents
    top_k_docs = set(
        sorted(retrieved.keys(), key=lambda x: retrieved[x], reverse=True)[:k]
    )

    # Count relevant documents in top-k
    relevant_in_topk = sum(
        1 for doc_id in top_k_docs if doc_id in relevant and relevant[doc_id] > 0
    )

    # Total number of relevant documents
    total_relevant = sum(1 for rel in relevant.values() if rel > 0)

    if total_relevant == 0:
        return 0.0
    return relevant_in_topk / total_relevant


def calculate_precision(
    retrieved: Dict[str, float], relevant: Dict[str, int], k: int
) -> float:
    """
    Calculate Precision@k.

    Args:
        retrieved: Dictionary mapping doc IDs to scores
        relevant: Dictionary mapping doc IDs to relevance labels
        k: Cutoff for evaluation

    Returns:
        Precision@k score
    """
    # Get top-k retrieved documents
    top_k_docs = sorted(retrieved.keys(), key=lambda x: retrieved[x], reverse=True)[:k]

    # Count relevant documents in top-k
    relevant_in_topk = sum(
        1 for doc_id in top_k_docs if doc_id in relevant and relevant[doc_id] > 0
    )

    if k == 0:
        return 0.0
    return relevant_in_topk / k


def calculate_mrr(retrieved: Dict[str, float], relevant: Dict[str, int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        retrieved: Dictionary mapping doc IDs to scores
        relevant: Dictionary mapping doc IDs to relevance labels

    Returns:
        MRR score
    """
    # Sort retrieved documents by score
    sorted_docs = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)

    # Find rank of first relevant document
    for i, (doc_id, _) in enumerate(sorted_docs):
        if doc_id in relevant and relevant[doc_id] > 0:
            return 1.0 / (i + 1)

    return 0.0


def calculate_f1(
    retrieved: Dict[str, float], relevant: Dict[str, int], k: int
) -> float:
    """
    Calculate F1@k score.

    Args:
        retrieved: Dictionary mapping doc IDs to scores
        relevant: Dictionary mapping doc IDs to relevance labels
        k: Cutoff for evaluation

    Returns:
        F1@k score
    """
    precision = calculate_precision(retrieved, relevant, k)
    recall = calculate_recall(retrieved, relevant, k)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_retrieval(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = [1, 3, 5, 10, 100],
    metrics: List[str] = ["ndcg", "map", "recall", "precision", "mrr", "f1"],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate retrieval results with multiple metrics.

    Args:
        results: Dictionary mapping query IDs to retrieved doc IDs and scores
        qrels: Dictionary mapping query IDs to relevant doc IDs and labels
        k_values: List of k values for metrics@k
        metrics: List of metrics to calculate

    Returns:
        Dictionary of metric scores
    """
    metric_scores: Dict[str, Dict[str, float]] = {}

    # Calculate each metric
    for metric_name in metrics:
        if metric_name == "map":
            # MAP doesn't use k
            scores = []
            for query_id in results:
                if query_id in qrels:
                    score = calculate_map(results[query_id], qrels[query_id])
                    scores.append(score)
            metric_scores["map"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }

        elif metric_name == "mrr":
            # MRR doesn't use k
            scores = []
            for query_id in results:
                if query_id in qrels:
                    score = calculate_mrr(results[query_id], qrels[query_id])
                    scores.append(score)
            metric_scores["mrr"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }

        else:
            # Metrics that use k
            for k in k_values:
                scores = []
                for query_id in results:
                    if query_id in qrels:
                        if metric_name == "ndcg":
                            score = calculate_ndcg(
                                results[query_id], qrels[query_id], k
                            )
                        elif metric_name == "recall":
                            score = calculate_recall(
                                results[query_id], qrels[query_id], k
                            )
                        elif metric_name == "precision":
                            score = calculate_precision(
                                results[query_id], qrels[query_id], k
                            )
                        elif metric_name == "f1":
                            score = calculate_f1(results[query_id], qrels[query_id], k)
                        else:
                            continue
                        scores.append(score)

                metric_scores[f"{metric_name}@{k}"] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                }

    return metric_scores


# Wrapper class for NDCG to match test expectations
class NDCG:
    """NDCG metric wrapper class."""

    def __init__(self, k: int = 10):
        """Initialize NDCG with k value.

        Args:
            k: Number of top results to consider
        """
        self.k = k

    def compute(
        self,
        retrieved: Union[Dict[str, float], torch.Tensor],
        relevant: Union[Dict[str, int], torch.Tensor],
    ) -> float:
        """
        Compute NDCG score.

        Args:
            retrieved: Dict mapping document IDs to scores or tensor of scores
            relevant: Dict mapping document IDs to relevance labels or tensor
                of relevance

        Returns:
            NDCG score
        """
        # Convert tensors to dictionaries if necessary
        if isinstance(retrieved, torch.Tensor) and isinstance(relevant, torch.Tensor):
            retrieved_dict = {
                f"doc_{i}": float(score) for i, score in enumerate(retrieved)
            }
            relevant_dict = {f"doc_{i}": int(rel) for i, rel in enumerate(relevant)}
            return calculate_ndcg(retrieved_dict, relevant_dict, self.k)
        elif isinstance(retrieved, dict) and isinstance(relevant, dict):
            return calculate_ndcg(retrieved, relevant, self.k)
        else:
            raise TypeError("Arguments must be both dicts or both tensors")

    def __call__(
        self,
        retrieved: Union[Dict[str, float], torch.Tensor],
        relevant: Union[Dict[str, int], torch.Tensor],
    ) -> float:
        """Allow calling the object directly."""
        return self.compute(retrieved, relevant)


# Function aliases for backward compatibility
def compute_mrr(
    retrieved: Union[Dict[str, float], torch.Tensor],
    relevant: Union[Dict[str, int], torch.Tensor],
) -> float:
    """Alias for calculate_mrr for backward compatibility."""
    # Convert tensors to dictionaries if necessary
    if isinstance(retrieved, torch.Tensor) and isinstance(relevant, torch.Tensor):
        retrieved_dict = {f"doc_{i}": float(score) for i, score in enumerate(retrieved)}
        relevant_dict = {f"doc_{i}": int(rel) for i, rel in enumerate(relevant)}
        return calculate_mrr(retrieved_dict, relevant_dict)
    elif isinstance(retrieved, dict) and isinstance(relevant, dict):
        return calculate_mrr(retrieved, relevant)
    else:
        raise TypeError("Arguments must be both dicts or both tensors")


def compute_recall_at_k(
    retrieved: Union[Dict[str, float], torch.Tensor],
    relevant: Union[Dict[str, int], torch.Tensor],
    k: int = 10,
) -> float:
    """Alias for calculate_recall for backward compatibility."""
    # Convert tensors to dictionaries if necessary
    if isinstance(retrieved, torch.Tensor) and isinstance(relevant, torch.Tensor):
        retrieved_dict = {f"doc_{i}": float(score) for i, score in enumerate(retrieved)}
        relevant_dict = {f"doc_{i}": int(rel) for i, rel in enumerate(relevant)}
        return calculate_recall(retrieved_dict, relevant_dict, k)
    elif isinstance(retrieved, dict) and isinstance(relevant, dict):
        return calculate_recall(retrieved, relevant, k)
    else:
        raise TypeError("Arguments must be both dicts or both tensors")
