"""Chamfer similarity computation for multi-vector representations."""

import torch
import torch.nn as nn


class ChamferSimilarity(nn.Module):
    """
    Compute Chamfer similarity between two sets of multi-vector
    representations.

    The Chamfer similarity is the average of:
    1. Maximum similarities from query to document vectors
    2. Maximum similarities from document to query vectors

    This provides a symmetric measure of similarity between two sets of vectors
    that is robust to different numbers of vectors in each set.
    """

    def __init__(self, symmetric: bool = True):
        """
        Initialize Chamfer similarity module.

        Args:
            symmetric: Whether to compute symmetric similarity (both
                directions)
        """
        super().__init__()
        self.symmetric = symmetric

    def forward(
        self,
        query_embeds: torch.Tensor,
        query_mask: torch.Tensor,
        doc_embeds: torch.Tensor,
        doc_mask: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute Chamfer similarity between query and document embeddings.

        Args:
            query_embeds: Query embeddings
                [batch_size, num_query_tokens, hidden_dim]
            query_mask: Query attention mask [batch_size, num_query_tokens]
            doc_embeds: Document embeddings
                [batch_size, num_doc_tokens, hidden_dim]
            doc_mask: Document attention mask [batch_size, num_doc_tokens]
            normalize: Whether to L2 normalize embeddings before similarity

        Returns:
            Similarity scores [batch_size, batch_size] or [batch_size] if not
                symmetric
        """
        # Normalize embeddings if requested
        if normalize:
            query_embeds = torch.nn.functional.normalize(query_embeds, p=2, dim=-1)
            doc_embeds = torch.nn.functional.normalize(doc_embeds, p=2, dim=-1)

        # Expand masks to match embedding dimensions
        query_mask = query_mask.unsqueeze(
            -1
        ).float()  # [batch_size, num_query_tokens, 1]
        doc_mask = doc_mask.unsqueeze(-1).float()
        # [batch_size, num_doc_tokens, 1]

        # Apply masks to embeddings
        query_embeds = query_embeds * query_mask
        doc_embeds = doc_embeds * doc_mask

        if self.symmetric:
            # Compute full batch similarity matrix
            similarities = self._compute_batch_similarities(
                query_embeds, query_mask, doc_embeds, doc_mask
            )
        else:
            # Compute only diagonal similarities
            similarities = self._compute_diagonal_similarities(
                query_embeds, query_mask, doc_embeds, doc_mask
            )

        return similarities

    def _compute_batch_similarities(
        self,
        query_embeds: torch.Tensor,
        query_mask: torch.Tensor,
        doc_embeds: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute full batch similarity matrix.

        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        # Reshape for batch computation
        # [batch_size, 1, num_query_tokens, hidden_dim]
        query_embeds_expanded = query_embeds.unsqueeze(1)
        # [1, batch_size, num_doc_tokens, hidden_dim]
        doc_embeds_expanded = doc_embeds.unsqueeze(0)

        # Compute all pairwise similarities
        # [batch_size, batch_size, num_query_tokens, num_doc_tokens]
        token_sims = torch.matmul(
            query_embeds_expanded, doc_embeds_expanded.transpose(-2, -1)
        )

        # Apply masks
        # query_mask: [batch_size, num_query_tokens, 1] ->
        # [batch_size, 1, num_query_tokens, 1]
        query_mask_expanded = query_mask.unsqueeze(1)
        # doc_mask: [batch_size, num_doc_tokens, 1] ->
        # [1, batch_size, 1, num_doc_tokens]
        doc_mask_expanded = doc_mask.permute(0, 2, 1).unsqueeze(0)

        # Create combined mask
        # [batch_size, batch_size, num_query_tokens, num_doc_tokens]
        mask = query_mask_expanded * doc_mask_expanded

        # Apply mask with large negative value for masked positions
        masked_sims = token_sims.masked_fill(mask == 0, -1e9)

        # Compute query->doc similarities (max over doc dimension)
        query_to_doc, _ = masked_sims.max(
            dim=-1
        )  # [batch_size, batch_size, num_query_tokens]

        # Mask out invalid similarities (those that are still -1e9)
        valid_query_mask = (query_to_doc > -1e8).float()

        # Sum only valid similarities and count valid tokens
        query_to_doc = torch.where(
            valid_query_mask > 0, query_to_doc, torch.zeros_like(query_to_doc)
        )
        valid_query_counts = valid_query_mask.sum(dim=-1).clamp(min=1)
        query_to_doc = query_to_doc.sum(dim=-1) / valid_query_counts
        # [batch_size, batch_size]

        # Compute doc->query similarities (max over query dimension)
        doc_to_query, _ = masked_sims.max(
            dim=-2
        )  # [batch_size, batch_size, num_doc_tokens]

        # Mask out invalid similarities (those that are still -1e9)
        valid_doc_mask = (doc_to_query > -1e8).float()

        # Sum only valid similarities and count valid tokens
        doc_to_query = torch.where(
            valid_doc_mask > 0, doc_to_query, torch.zeros_like(doc_to_query)
        )
        valid_doc_counts = valid_doc_mask.sum(dim=-1).clamp(min=1)
        doc_to_query = doc_to_query.sum(dim=-1) / valid_doc_counts
        # [batch_size, batch_size]

        # Average both directions
        similarities = (query_to_doc + doc_to_query) / 2.0

        return similarities

    def _compute_diagonal_similarities(
        self,
        query_embeds: torch.Tensor,
        query_mask: torch.Tensor,
        doc_embeds: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute only diagonal similarities (more efficient for training).

        Returns:
            Similarity scores [batch_size]
        """
        # Compute token-level similarities
        # [batch_size, num_query_tokens, num_doc_tokens]
        token_sims = torch.bmm(query_embeds, doc_embeds.transpose(1, 2))

        # Create mask
        mask = torch.bmm(query_mask, doc_mask.transpose(1, 2))
        # [batch_size, num_query_tokens, num_doc_tokens]

        # Apply mask
        masked_sims = token_sims.masked_fill(mask == 0, -1e9)

        # Query->Doc: max over doc dimension
        query_to_doc, _ = masked_sims.max(dim=-1)
        # [batch_size, num_query_tokens]

        # Mask out invalid similarities
        valid_query_mask = (query_to_doc > -1e8).float()
        query_to_doc = torch.where(
            valid_query_mask > 0, query_to_doc, torch.zeros_like(query_to_doc)
        )
        valid_query_counts = valid_query_mask.sum(dim=-1).clamp(min=1)
        query_to_doc = query_to_doc.sum(dim=-1) / valid_query_counts
        # [batch_size]

        # Doc->Query: max over query dimension
        doc_to_query, _ = masked_sims.max(dim=1)
        # [batch_size, num_doc_tokens]

        # Mask out invalid similarities
        valid_doc_mask = (doc_to_query > -1e8).float()
        doc_to_query = torch.where(
            valid_doc_mask > 0, doc_to_query, torch.zeros_like(doc_to_query)
        )
        valid_doc_counts = valid_doc_mask.sum(dim=-1).clamp(min=1)
        doc_to_query = doc_to_query.sum(dim=-1) / valid_doc_counts
        # [batch_size]

        # Average both directions
        similarities = (query_to_doc + doc_to_query) / 2.0

        return similarities

    @staticmethod
    def compute_retrieval_scores(
        query_embeds: torch.Tensor,
        query_mask: torch.Tensor,
        doc_embeds: torch.Tensor,
        doc_mask: torch.Tensor,
        batch_size: int = 128,
    ) -> torch.Tensor:
        """
        Compute retrieval scores for large document collections.

        This static method is useful for evaluation where we need to score
        many documents against queries efficiently.

        Args:
            query_embeds: Query embeddings
                [num_queries, num_tokens, hidden_dim]
            query_mask: Query masks [num_queries, num_tokens]
            doc_embeds: Document embeddings
                [num_docs, num_tokens, hidden_dim]
            doc_mask: Document masks [num_docs, num_tokens]
            batch_size: Batch size for processing

        Returns:
            Similarity scores [num_queries, num_docs]
        """
        num_queries = query_embeds.size(0)
        num_docs = doc_embeds.size(0)

        # Initialize similarity matrix
        similarities = torch.zeros(num_queries, num_docs, device=query_embeds.device)

        # Create similarity module
        chamfer = ChamferSimilarity(symmetric=True)

        # Process in batches to manage memory
        for i in range(0, num_queries, batch_size):
            end_i = min(i + batch_size, num_queries)
            query_batch = query_embeds[i:end_i]
            query_mask_batch = query_mask[i:end_i]

            for j in range(0, num_docs, batch_size):
                end_j = min(j + batch_size, num_docs)
                doc_batch = doc_embeds[j:end_j]
                doc_mask_batch = doc_mask[j:end_j]

                # Compute similarities for this batch
                batch_sims = chamfer(
                    query_batch, query_mask_batch, doc_batch, doc_mask_batch
                )

                similarities[i:end_i, j:end_j] = batch_sims

        return similarities
