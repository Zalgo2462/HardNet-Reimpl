import torch


class LossHardNet:
    def __init__(self, margin):
        """
        TODO: finish commenting
        :param margin:
        """
        super().__init__()
        self.margin = margin

    def impl(self, anchor, positive):
        """
        type: (LossHardNet, Variable, Variable)-> Tensor
        TODO: finish commenting
        :return:
        """
        pass

    @staticmethod
    def _get_negative(anchor, positive):
        """
        type: (LossHardNet, Variable, Variable)-> Variable
        TODO: finish commenting
        :param anchor: NXD matrix of N D dimensional descriptors
        :param positive: NXD matrix of N D dimensional descriptors
        :return:
        """

        assert anchor.size() == positive.size()
        assert anchor.dim() == 2


        distance_matrix =  LossHardNet.__all_pairs_distance_matrix(anchor, positive) # + eps?

        pass

    @staticmethod
    def __all_pairs_distance_matrix(anchor, positive):
        """
        type: (LossHardNet, Variable, Variable)-> Variable

        Calculates the pairwise distance between each row vector in anchor and each row vector
        in positive. Stores the distances in a symmetric matrix.

        :param anchor: NXD matrix of N D dimensional descriptors
        :param positive: NXD matrix of N D dimensional descriptors
        :return: NXN symmetric matrix of pairwise distance calculations
        """

        # Use the fact that ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*<x1,x2>

        anchor_norm_sq = torch.sum(anchor * anchor, dim=(1,)).unsqueeze(-1).repeat((1, anchor.size(0)))
        positive_norm_sq = torch.t(torch.sum(anchor * anchor, dim=(1,)).unsqueeze(-1).repeat((1, anchor.size(0))))

        return torch.sqrt(anchor_norm_sq + positive_norm_sq - 2.0 * torch.mm(anchor, torch.t(positive)))

    @staticmethod
    def __filter_distance_matrix(distances):
        """
        type: (Variable)->Variable
        TODO: commenting and implementation
        :param distances:
        :return:
        """
        pass
