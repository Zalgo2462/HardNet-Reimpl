import torch
from torch import Tensor

from LossHardNet import LossHardNet


class LossHardNetContrastive(LossHardNet):
    def __init__(self, margin):
        # type:(LossHardNetContrastive, float) -> None
        """
        Implements a constrastive loss using intra batch
        negative mining.

        :param margin: Contrastive loss margin
        """
        super().__init__()
        self.__margin = margin

    def impl(self, anchor, positive):
        # type: (LossHardNetContrastive, Tensor, Tensor) -> Tensor
        """
        Implements a constrastive loss using intra batch
        negative mining.

        :return: a loss value for the batch
        """
        positive_distances, negative_distances = LossHardNet._get_positive_negative_distances(anchor, positive)

        # loss = torch.clamp(self.__margin - negative_distances, min=0.0) + positive_distances;
        # loss = torch.mean(loss)

        return torch.mean(
            torch.clamp(
                torch.add(
                    torch.mul(
                        negative_distances,
                        -1
                    ),
                    self.__margin
                ),
                min=0.0
            ) + positive_distances
        )
