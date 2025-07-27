"""Contrastive loss implementation for CRISP training."""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.

    This loss encourages positive pairs to have high similarity while
    pushing negative pairs to have low similarity. It's particularly
    effective for retrieval tasks.

    The loss is computed as:
    L = -log(exp(sim(q, d+) / tau) / sum(exp(sim(q, di) / tau)))

    where:
    - q is the query
    - d+ is the positive document
    - di are all documents in the batch
    - tau is the temperature parameter
    """

    def __init__(
        self,
        temperature: float = 0.05,
        scale_by_temperature: bool = True,
        reduction: str = "mean",
    ):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for scaling similarities
            scale_by_temperature: Whether to scale similarities by temperature
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.reduction = reduction

    def forward(
        self,
        similarities: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            similarities: Similarity matrix [batch_size, batch_size]
                         or [batch_size] for diagonal only
            labels: Optional labels indicating positive pairs
                    If None, assumes diagonal elements are positives
            mask: Optional mask for valid pairs

        Returns:
            Loss value (scalar if reduction is 'mean' or 'sum')
        """
        device = similarities.device

        # Handle different input shapes
        if similarities.dim() == 1:
            # Convert diagonal similarities to full matrix
            batch_size = similarities.size(0)
            sim_matrix = torch.zeros(batch_size, batch_size, device=device)
            sim_matrix.diagonal().copy_(similarities)
            similarities = sim_matrix

        batch_size = similarities.size(0)

        # Handle single sample case
        if batch_size == 1:
            # With only one sample, there are no negatives for contrastive learning
            # Return a small positive loss to indicate this edge case
            return torch.tensor(1e-6, device=device)

        # Create default labels if not provided (diagonal as positives)
        if labels is None:
            labels = torch.arange(batch_size, device=device)

        # Scale by temperature if requested
        if self.scale_by_temperature:
            similarities = similarities / self.temperature

        # Create mask for valid negatives if not provided
        if mask is None:
            mask = torch.ones_like(similarities, dtype=torch.bool)
            # Mask out diagonal for standard contrastive learning
            mask.fill_diagonal_(False)

        # Compute log softmax along each row
        # This handles numerical stability automatically
        log_probs = F.log_softmax(similarities, dim=1)

        # Select log probabilities for positive pairs
        if labels.dim() == 1 and labels.size(0) == batch_size:
            # Standard case: one positive per query
            positive_log_probs = log_probs[torch.arange(batch_size), labels]
            loss = -positive_log_probs
        else:
            # Multiple positives per query
            raise NotImplementedError("Multiple positives per query not yet supported")

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    @staticmethod
    def distributed_gather_similarities(
        local_similarities: torch.Tensor, world_size: int, rank: int
    ) -> torch.Tensor:
        """
        Gather similarities from all processes in distributed training.

        This is useful for computing loss across all pairs in the
        global batch when using data parallel training.

        Args:
            local_similarities: Local similarity matrix
            world_size: Number of processes
            rank: Current process rank

        Returns:
            Global similarity matrix
        """
        if world_size == 1:
            return local_similarities

        # Gather all similarities
        gathered_similarities = [
            torch.zeros_like(local_similarities) for _ in range(world_size)
        ]
        dist.all_gather(gathered_similarities, local_similarities)

        # Concatenate to form global matrix
        global_similarities = torch.cat(gathered_similarities, dim=0)

        return global_similarities


class TripletLoss(nn.Module):
    """
    Triplet loss for learning embeddings.

    This loss ensures that the distance between anchor and positive
    is smaller than the distance between anchor and negative by a margin.
    """

    def __init__(
        self,
        margin: float = 0.1,
        distance_metric: str = "euclidean",
        reduction: str = "mean",
    ):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
            distance_metric: Distance metric ('euclidean' or 'cosine')
            reduction: Reduction method
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.reduction = reduction

    def forward(
        self,
        anchor_embeds: torch.Tensor,
        positive_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
        anchor_mask: Optional[torch.Tensor] = None,
        positive_mask: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor_embeds: Anchor embeddings
            positive_embeds: Positive embeddings
            negative_embeds: Negative embeddings
            *_mask: Optional attention masks

        Returns:
            Loss value
        """
        # Compute distances
        if self.distance_metric == "euclidean":
            pos_dist = self._euclidean_distance(
                anchor_embeds, positive_embeds, anchor_mask, positive_mask
            )
            neg_dist = self._euclidean_distance(
                anchor_embeds, negative_embeds, anchor_mask, negative_mask
            )
        else:  # cosine
            pos_dist = 1 - self._cosine_similarity(
                anchor_embeds, positive_embeds, anchor_mask, positive_mask
            )
            neg_dist = 1 - self._cosine_similarity(
                anchor_embeds, negative_embeds, anchor_mask, negative_mask
            )

        # Compute triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def _euclidean_distance(
        self,
        embeds1: torch.Tensor,
        embeds2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Euclidean distance between embeddings."""
        if embeds1.dim() == 3:  # Multi-vector case
            # Apply masks if provided
            if mask1 is not None:
                embeds1 = embeds1 * mask1.unsqueeze(-1)
            if mask2 is not None:
                embeds2 = embeds2 * mask2.unsqueeze(-1)

            # Average pool over sequence dimension
            embeds1 = embeds1.mean(dim=1)
            embeds2 = embeds2.mean(dim=1)

        distance: torch.Tensor = torch.norm(embeds1 - embeds2, p=2, dim=-1)
        return distance

    def _cosine_similarity(
        self,
        embeds1: torch.Tensor,
        embeds2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cosine similarity between embeddings."""
        if embeds1.dim() == 3:  # Multi-vector case
            # Apply masks if provided
            if mask1 is not None:
                embeds1 = embeds1 * mask1.unsqueeze(-1)
            if mask2 is not None:
                embeds2 = embeds2 * mask2.unsqueeze(-1)

            # Average pool over sequence dimension
            embeds1 = embeds1.mean(dim=1)
            embeds2 = embeds2.mean(dim=1)

        # Normalize embeddings
        embeds1 = F.normalize(embeds1, p=2, dim=-1)
        embeds2 = F.normalize(embeds2, p=2, dim=-1)

        return (embeds1 * embeds2).sum(dim=-1)
