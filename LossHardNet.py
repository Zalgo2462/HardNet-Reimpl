from typing import Tuple

import torch
import torch.cuda
from torch import Tensor


class LossHardNet:
    DUPLICATE_THRESHOLD = 0.008

    def __init__(self):
        # type: (LossHardNet) -> None
        """
        Implements a HardNet style loss with intra batch negative mining

        :param margin: Loss margin
        """
        super().__init__()

    def impl(self, anchor, positive):
        # type: (LossHardNet, Tensor, Tensor)-> Tensor
        """
        Calculates the loss for a given batch of HardNet inputs
        :return: A Tensor containing the loss for the batch
        """
        pass

    @staticmethod
    def _get_positive_negative_distances(anchor, positive):
        # type: (Tensor, Tensor)-> Tuple[Tensor, Tensor]
        """
        Finds the hardest negative in the given batch using the "min" selection
        strategy from the original HardNet implementation

        :param anchor: NxD matrix of N D dimensional descriptors
        :param positive: NxD matrix of N D dimensional descriptors
        :return: 1xN matrix of distances from each anchor to its closest negative
        """

        assert anchor.size() == positive.size()
        assert anchor.dim() == 2

        distance_matrix = LossHardNet.__all_pairs_distance_matrix(anchor, positive)  # + eps?
        filtered_distance_matrix = LossHardNet.__remove_false_negatives(distance_matrix)

        # Apply the "min" selection from the original implementation with anchor swapping
        closest_positive_distances = torch.min(filtered_distance_matrix, 1)[0]
        closest_anchor_distances = torch.min(filtered_distance_matrix, 0)[0]
        closest_negative_distances = torch.min(closest_positive_distances, closest_anchor_distances)

        return torch.diag(distance_matrix), closest_negative_distances

    @staticmethod
    def __all_pairs_distance_matrix(anchor, positive):
        # type: (Tensor, Tensor)-> Tensor
        """
        Calculates the pairwise distance between each row vector in anchor and each row vector
        in positive. Stores the distances in a symmetric matrix.

        :param anchor: NxD matrix of N D dimensional descriptors
        :param positive: NxD matrix of N D dimensional descriptors
        :return: NxN symmetric matrix of pairwise distance calculations
                 with anchor indexes along the rows and positive indexes along
                 the columns.
        """

        # Use the fact that ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*<x1,x2>

        anchor_norm_sq = torch.sum(
            torch.mul(anchor, anchor),
            dim=(1,)
        ).unsqueeze(-1).repeat((1, anchor.size(0)))

        positive_norm_sq = torch.t(torch.sum(
            torch.mul(anchor, anchor),
            dim=(1,)
        ).unsqueeze(-1).repeat((1, anchor.size(0))))

        return torch.sqrt(
            torch.add(
                torch.add(anchor_norm_sq, positive_norm_sq),
                torch.mul(
                    torch.mm(anchor, torch.t(positive)), -2.0
                )
            )
        )

    @staticmethod
    def __remove_false_negatives(distances):
        # type: (Tensor) -> Tensor
        """
        Filters the symmetric anchor-positive distance matrix for negative
        selection. Ensures the negative selector does not select a positive
        match or a duplicate match.

        :param distances: A symmetric anchor-positive distance matrix with anchor
                          anchor indexes along the rows and positive indexes along
                          the columns.
        :return: filtered distances for negative selection
        """

        # Ensure we don't select the positives matches as negatives
        identity_matrix = torch.eye(distances.size(1), requires_grad=True)

        # TODO: Add support for disabling cuda
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()

        filtered = torch.add(
            distances,
            torch.mul(
                identity_matrix,
                10
            )
        )

        # Anchor patches may appear as each other's positive matches
        # Filter out the duplicates
        # (Arguably this could be handled when the batch is created)

        duplicates = filtered.lt(LossHardNet.DUPLICATE_THRESHOLD)

        filtered = torch.add(
            filtered,
            torch.mul(
                duplicates.float(),
                10
            )
        )

        return filtered
