import torch
from torch import Tensor

from LossHardNet import LossHardNet


class LossHardNetTripletMargin(LossHardNet):
    def __init__(self, margin):
        # type: (LossHardNetTripletMargin, float) -> None
        """
        Implements a triplet loss using intra batch
        negative mining.

        :param margin: Triplet loss margin
        """
        super().__init__()
        self.__margin = margin

    def impl(self, anchor, positive):
        # type: (LossHardNetTripletMargin, Tensor, Tensor) -> Tensor
        """
        Implements a triplet margin loss using intra batch negative mining.

        :return: a loss value for the batch
        """

        positive_distances, negative_distances = LossHardNet._get_positive_negative_distances(anchor, positive)

        # loss = torch.clamp(self.__margin + positive_distances - min_distances, min=0.0)
        # loss = torch.mean(loss)
        return torch.mean(
            torch.clamp(
                torch.add(
                    torch.add(
                        positive_distances,
                        torch.mul(
                            negative_distances,
                            -1
                        )
                    ),
                    self.__margin
                ),
                min=0.0
            )
        )
