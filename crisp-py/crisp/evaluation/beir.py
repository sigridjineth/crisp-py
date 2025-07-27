"""BEIR evaluation for CRISP models."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from ..clustering import create_clustering_strategy
from ..config import CRISPConfig
from ..data import load_beir_dataset
from ..losses.chamfer import ChamferSimilarity
from ..models import CRISPEncoder
from ..pruning import create_pruning_strategy
from .metrics import calculate_map, calculate_ndcg, calculate_recall

logger = logging.getLogger(__name__)


class BEIREvaluator:
    """
    Evaluator for CRISP models on BEIR benchmarks.

    Handles encoding, clustering/pruning, and evaluation of retrieval
    performance on BEIR datasets.

    Attributes:
        encoder: CRISP encoder model
        clustering_strategy: Optional clustering strategy
        pruning_strategy: Optional pruning strategy
        device: Device for computation
        batch_size: Batch size for encoding
    """

    def __init__(
        self,
        encoder: CRISPEncoder,
        config: CRISPConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize BEIR evaluator.

        Args:
            encoder: CRISP encoder model
            config: CRISP configuration
            device: Device for computation
        """
        self.model = encoder  # Store as 'model' for test compatibility
        self.encoder = encoder
        self.config = config
        self.use_instruction_prefix = config.use_instruction_prefix

        # Create pruning or clustering strategy from config
        self.pruning_strategy = create_pruning_strategy(config)
        self.clustering_strategy = create_clustering_strategy(config)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = config.batch_size

        # Add tokenizer attribute for test compatibility
        self.tokenizer = encoder.tokenizer if hasattr(encoder, "tokenizer") else None

        # Move encoder to device
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

    def evaluate(
        self,
        dataset_name: Optional[str] = None,
        data_path: Optional[Union[str, Path]] = None,
        corpus: Optional[Dict[str, Dict[str, str]]] = None,
        queries: Optional[Dict[str, str]] = None,
        qrels: Optional[Dict[str, Dict[str, int]]] = None,
        k_values: List[int] = [1, 3, 5, 10, 100],
        save_results: bool = True,
        results_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluate on a BEIR dataset.

        Args:
            dataset_name: Name of BEIR dataset (optional if
                corpus/queries/qrels provided)
            data_path: Path to BEIR data (optional if corpus/queries/qrels
                provided)
            corpus: Document corpus (optional, loaded from dataset if not
                provided)
            queries: Query dictionary (optional, loaded from dataset if not
                provided)
            qrels: Relevance judgments (optional, loaded from dataset if not
                provided)
            k_values: List of k values for metrics@k
            save_results: Whether to save results
            results_dir: Directory to save results

        Returns:
            Dictionary of evaluation metrics
        """
        # Support both dataset loading and direct data passing (for tests)
        if corpus is None or queries is None or qrels is None:
            if dataset_name is None or data_path is None:
                raise ValueError(
                    "Either provide dataset_name and data_path, or provide "
                    "corpus, queries, and qrels"
                )

            logger.info(f"Evaluating on {dataset_name}")
            # Load dataset
            dataset = load_beir_dataset(dataset_name, str(data_path))
            queries = dataset.get_queries()
            corpus = dataset.get_corpus()  # type: ignore
            qrels = dataset.get_qrels()
        else:
            logger.info("Evaluating on provided data")

        # Ensure we have all required data
        if not corpus or not queries or not qrels:
            raise ValueError("corpus, queries, and qrels must all be provided")

        # Encode queries and documents
        logger.info("Encoding queries...")
        query_embeddings = self._encode_texts(
            list(queries.values()),
            list(queries.keys()),
            is_query=True,
            desc="Encoding queries",
        )

        logger.info("Encoding documents...")
        # Handle both corpus formats: Dict[str, str] or
        # Dict[str, Dict[str, str]]
        if corpus and isinstance(next(iter(corpus.values())), dict):
            # Extract text from nested dict format
            doc_texts = [
                doc.get("text", doc.get("title", ""))
                for doc in corpus.values()  # type: ignore
            ]
        else:
            # Simple string format
            doc_texts = list(corpus.values())  # type: ignore

        doc_embeddings = self._encode_texts(
            doc_texts,
            list(corpus.keys()),
            is_query=False,
            desc="Encoding documents",
        )

        # Calculate similarities and retrieve
        logger.info("Computing similarities...")
        results = self._retrieve(
            query_embeddings,
            doc_embeddings,
            list(queries.keys()),
            list(corpus.keys()),
            top_k=max(k_values),
        )

        # Evaluate metrics
        metrics = self._evaluate_metrics(results, qrels, k_values)

        # Save results if requested
        if save_results and results_dir and dataset_name:
            self._save_results(dataset_name, metrics, results_dir)

        return metrics

    def _encode_texts(
        self,
        texts: List[str],
        ids: List[str],
        is_query: bool = True,
        desc: str = "Encoding",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Encode a list of texts.

        Args:
            texts: List of text strings
            ids: List of text IDs
            is_query: Whether encoding queries (vs documents)
            desc: Description for progress bar

        Returns:
            Dictionary mapping IDs to embeddings
        """
        embeddings_dict = {}

        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc=desc):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids = ids[i : i + self.batch_size]

            # Encode batch
            with torch.no_grad():
                batch_embeddings = self.encoder.encode(
                    batch_texts,
                    is_query=is_query,
                    batch_size=len(batch_texts),
                    device=self.device,
                    show_progress=False,
                )

            # Process embeddings (clustering/pruning)
            processed_embeddings = self._process_embeddings(
                batch_embeddings["embeddings"],
                batch_embeddings["attention_mask"],
                is_query=is_query,
            )

            # Store in dictionary
            for j, text_id in enumerate(batch_ids):
                embeddings_dict[text_id] = {
                    "embeddings": processed_embeddings["embeddings"][j],
                    "attention_mask": (processed_embeddings["attention_mask"][j]),
                }

        return embeddings_dict

    def _process_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        is_query: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply clustering or pruning to embeddings.

        Args:
            embeddings: Token embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            is_query: Whether these are query embeddings

        Returns:
            Processed embeddings and updated masks
        """
        # Apply pruning if configured
        if self.pruning_strategy is not None:
            embeddings = self.pruning_strategy.prune(
                embeddings, attention_mask, is_query
            )
            # Update attention mask to match new embedding size
            new_seq_len = embeddings.size(1)
            attention_mask = torch.ones(
                embeddings.size(0), new_seq_len, device=embeddings.device
            )

        # Apply clustering if configured
        elif self.clustering_strategy is not None:
            embeddings = self.clustering_strategy.cluster(
                embeddings, attention_mask, is_query
            )
            # Update attention mask to match new embedding size
            new_seq_len = embeddings.size(1)
            attention_mask = torch.ones(
                embeddings.size(0), new_seq_len, device=embeddings.device
            )

        return {"embeddings": embeddings, "attention_mask": attention_mask}

    def _retrieve(
        self,
        query_embeddings: Dict[str, Dict[str, torch.Tensor]],
        doc_embeddings: Dict[str, Dict[str, torch.Tensor]],
        query_ids: List[str],
        doc_ids: List[str],
        top_k: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieve top-k documents for each query using Chamfer similarity.

        Args:
            query_embeddings: Query embeddings dictionary
            doc_embeddings: Document embeddings dictionary
            query_ids: List of query IDs
            doc_ids: List of document IDs
            top_k: Number of documents to retrieve

        Returns:
            Dictionary mapping query IDs to retrieved doc IDs and scores
        """
        results = {}

        for query_id in tqdm(query_ids, desc="Retrieving"):
            q_emb = query_embeddings[query_id]["embeddings"]
            q_mask = query_embeddings[query_id]["attention_mask"]

            # Calculate similarities with all documents
            scores = []
            for doc_id in doc_ids:
                d_emb = doc_embeddings[doc_id]["embeddings"]
                d_mask = doc_embeddings[doc_id]["attention_mask"]

                # Compute Chamfer similarity
                score = self._chamfer_similarity(q_emb, q_mask, d_emb, d_mask)
                scores.append(score)

            # Get top-k documents
            scores_array = np.array(scores)
            top_indices = np.argsort(scores_array)[::-1][:top_k]

            # Store results
            results[query_id] = {
                doc_ids[idx]: float(scores_array[idx]) for idx in top_indices
            }

        return results

    def _chamfer_similarity(
        self,
        query_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_embeddings: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> float:
        """
        Compute Chamfer similarity between query and document.

        Args:
            query_embeddings: Query embeddings [seq_len, hidden_dim]
            query_mask: Query attention mask [seq_len]
            doc_embeddings: Document embeddings [seq_len, hidden_dim]
            doc_mask: Document attention mask [seq_len]

        Returns:
            Chamfer similarity score
        """
        # Move to device if needed
        if query_embeddings.device != self.device:
            query_embeddings = query_embeddings.to(self.device)
            query_mask = query_mask.to(self.device)
            doc_embeddings = doc_embeddings.to(self.device)
            doc_mask = doc_mask.to(self.device)

        # Apply masks
        q_emb = query_embeddings[query_mask.bool()]
        # [num_q_tokens, hidden_dim]
        d_emb = doc_embeddings[doc_mask.bool()]  # [num_d_tokens, hidden_dim]

        if len(q_emb) == 0 or len(d_emb) == 0:
            return 0.0

        # Compute similarity matrix
        sim_matrix = torch.matmul(q_emb, d_emb.T)
        # [num_q_tokens, num_d_tokens]

        # Chamfer similarity: sum of max similarities for each query token
        chamfer_score = sim_matrix.max(dim=1)[0].sum().item()

        return chamfer_score

    def _evaluate_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int],
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate evaluation metrics.

        Args:
            results: Retrieved results
            qrels: Relevance judgments
            k_values: List of k values

        Returns:
            Dictionary of metrics
        """
        metrics: Dict[str, Union[float, Dict[str, float]]] = {}

        # Calculate NDCG@k
        for k in k_values:
            ndcg_scores = []
            for query_id in results:
                if query_id in qrels:
                    ndcg = calculate_ndcg(results[query_id], qrels[query_id], k)
                    ndcg_scores.append(ndcg)

            if ndcg_scores:
                metrics[f"ndcg@{k}"] = {
                    "mean": float(np.mean(ndcg_scores)),
                    "std": float(np.std(ndcg_scores)),
                }
            else:
                metrics[f"ndcg@{k}"] = {"mean": 0.0, "std": 0.0}  # type: ignore[dict-item]

        # Add simplified ndcg@10 for test compatibility
        if "ndcg@10" in metrics and isinstance(metrics["ndcg@10"], dict):
            metrics["ndcg@10"] = float(metrics["ndcg@10"]["mean"])  # type: ignore

        # Calculate MAP
        map_scores = []
        for query_id in results:
            if query_id in qrels:
                map_score = calculate_map(results[query_id], qrels[query_id])
                map_scores.append(map_score)

        metrics["map"] = {
            "mean": float(np.mean(map_scores)),
            "std": float(np.std(map_scores)),
        }  # type: ignore[dict-item]

        # Calculate Recall@k
        for k in k_values:
            recall_scores = []
            for query_id in results:
                if query_id in qrels:
                    recall = calculate_recall(results[query_id], qrels[query_id], k)
                    recall_scores.append(recall)

            metrics[f"recall@{k}"] = {
                "mean": float(np.mean(recall_scores)),
                "std": float(np.std(recall_scores)),
            }  # type: ignore[dict-item]

        return metrics

    def _save_results(
        self,
        dataset_name: str,
        metrics: Dict[str, Union[float, Dict[str, float]]],
        results_dir: Union[str, Path],
    ):
        """Save evaluation results to file."""
        import json

        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with strategy info
        strategy_suffix = ""
        if self.clustering_strategy:
            strategy_suffix += f"_{self.clustering_strategy.__class__.__name__}"
        if self.pruning_strategy:
            strategy_suffix += f"_{self.pruning_strategy.__class__.__name__}"

        filename = results_dir / f"{dataset_name}{strategy_suffix}_results.json"

        # Save metrics
        with open(filename, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Results saved to {filename}")

    def compute_similarities(
        self,
        queries: List[str],
        documents: List[str],
        tokenizer=None,
    ) -> torch.Tensor:
        """
        Compute similarities between queries and documents.

        This method is for test compatibility - it provides a simpler interface
        for computing similarities without the full BEIR evaluation pipeline.

        Args:
            queries: List of query strings
            documents: List of document strings
            tokenizer: Optional tokenizer (not used, for compatibility)

        Returns:
            Similarity matrix [num_queries, num_docs]
        """
        # Encode queries
        with torch.no_grad():
            query_outputs = self.encoder.encode(
                queries,
                is_query=True,
                batch_size=self.batch_size,
                device=self.device,
                show_progress=False,
            )

        # Encode documents
        with torch.no_grad():
            doc_outputs = self.encoder.encode(
                documents,
                is_query=False,
                batch_size=self.batch_size,
                device=self.device,
                show_progress=False,
            )

        # Process embeddings if needed
        query_processed = self._process_embeddings(
            query_outputs["embeddings"],
            query_outputs["attention_mask"],
            is_query=True,
        )

        doc_processed = self._process_embeddings(
            doc_outputs["embeddings"],
            doc_outputs["attention_mask"],
            is_query=False,
        )

        # Use ChamferSimilarity to compute scores
        scores = ChamferSimilarity.compute_retrieval_scores(
            query_processed["embeddings"],
            query_processed["attention_mask"],
            doc_processed["embeddings"],
            doc_processed["attention_mask"],
            batch_size=self.batch_size,
        )

        return scores

    def _encode_queries(
        self, queries: Dict[str, str], tokenizer=None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Encode queries for test compatibility.

        Args:
            queries: Dictionary of query_id -> query_text
            tokenizer: Optional tokenizer (uses self.tokenizer if not provided)

        Returns:
            Dictionary of query_id -> embeddings
        """
        # Use instruction prefix if configured
        if self.use_instruction_prefix and hasattr(self.config, "beir_task"):
            instruction = (
                get_beir_instruction(self.config.beir_task)
                if self.config.beir_task
                else ""
            )
            if instruction:
                # Prepend instruction to queries
                queries = {
                    qid: f"{instruction} | query: {qtext}"
                    for qid, qtext in queries.items()
                }

        # Encode using the standard method
        return self._encode_texts(
            list(queries.values()),
            list(queries.keys()),
            is_query=True,
            desc="Encoding queries",
        )

    def _create_attention_masks(self, embeddings, tokenizer=None):
        """
        Create attention masks for embeddings.

        This method creates masks by detecting zero vectors (padding).

        Args:
            embeddings: Tensor of embeddings
            tokenizer: Optional tokenizer (not used, for compatibility)

        Returns:
            Attention mask tensor
        """
        # Detect non-zero vectors as valid tokens
        if embeddings.dim() == 3:
            # Sum across embedding dimension to check if vector is all zeros
            embedding_norms = embeddings.norm(dim=-1)  # [batch_size, seq_len]
            # Create mask: 1 for non-zero vectors, 0 for zero vectors
            attention_mask = (embedding_norms > 1e-6).float()
            return attention_mask
        else:
            return torch.ones_like(embeddings[..., 0])


def evaluate_on_beir(
    model: CRISPEncoder,
    dataset_names: List[str],
    data_path: Union[str, Path],
    config: CRISPConfig,
    k_values: List[int] = [1, 3, 5, 10, 100],
    save_results: bool = True,
    results_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """
    Evaluate a CRISP model on multiple BEIR datasets.

    Args:
        model: CRISP encoder model
        dataset_names: List of BEIR dataset names
        data_path: Path to BEIR data directory
        clustering_strategy: Optional clustering strategy
        pruning_strategy: Optional pruning strategy
        batch_size: Batch size for encoding
        k_values: List of k values for metrics@k
        save_results: Whether to save results
        results_dir: Directory to save results

    Returns:
        Dictionary mapping dataset names to metrics
    """
    evaluator = BEIREvaluator(
        encoder=model,
        config=config,
    )

    all_results = {}

    for dataset_name in dataset_names:
        try:
            results = evaluator.evaluate(
                dataset_name=dataset_name,
                data_path=data_path,
                k_values=k_values,
                save_results=save_results,
                results_dir=results_dir,
            )
            all_results[dataset_name] = results

        except Exception as e:
            logger.error(f"Error evaluating on {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}  # type: ignore

    return all_results


def get_beir_instruction(task_name: str) -> str:
    """
    Get task-specific instruction for BEIR datasets.

    Args:
        task_name: Name of the BEIR task

    Returns:
        Task-specific instruction string, empty string if task unknown
    """
    instructions = {
        "msmarco": (
            "Given a web search query, retrieve relevant passages that "
            "answer the query"
        ),
        "trec-covid": (
            "Given a query about COVID-19, retrieve relevant scientific " "articles"
        ),
        "nfcorpus": (
            "Given a biomedical query, retrieve relevant documents from the "
            "NutritionFacts corpus"
        ),
        "bioasq": ("Given a biomedical question, retrieve relevant PubMed abstracts"),
        "scidocs": "Given a scientific paper title, retrieve cited papers",
        "arguana": (
            "Given an argument, retrieve counter-arguments from debate " "portals"
        ),
        "touche-2020": (
            "Given a comparative question, retrieve argumentative passages"
        ),
        "webis-touche2020": (
            "Given a comparative question, retrieve argumentative passages"
        ),
        "quora": "Given a question, retrieve duplicate questions from Quora",
        "dbpedia-entity": (
            "Given an entity name, retrieve relevant Wikipedia passages"
        ),
        "scifact": ("Given a scientific claim, retrieve evidence from research papers"),
        "fever": "Given a factual claim, retrieve evidence from Wikipedia",
        "climate-fever": (
            "Given a climate-related claim, retrieve evidence from Wikipedia"
        ),
        "hotpotqa": (
            "Given a multi-hop question, retrieve supporting facts from " "Wikipedia"
        ),
        "fiqa": ("Given a financial question, retrieve relevant answers from forums"),
        "nq": (
            "Given a question, retrieve Wikipedia passages containing the " "answer"
        ),
        "cqadupstack": (
            "Given a technical question, retrieve duplicate questions from "
            "Stack Exchange"
        ),
    }

    return instructions.get(task_name.lower(), "")
