from abc import ABC, abstractmethod

import torch


class PruningStrategy(ABC):

    @abstractmethod
    def prune(
        self, embeddings: torch.Tensor, mask: torch.Tensor, is_query: bool
    ) -> torch.Tensor:
        # subclasses implement this
        pass

    def get_output_size(self, input_size: int, is_query: bool) -> int:
        # can be overridden if size is known
        return -1
