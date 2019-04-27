from LossHardNet import LossHardNet


class LossHardNetContrastive(LossHardNet):
    def __init__(self, margin):
        """
        TODO: finish commenting and implement
        :param margin:
        """
        super().__init__(margin)

    def impl(self, anchor, positive):
        """
        type: (Loss, Variable, Variable)-> Tensor
        TODO: finish commenting and implement
        :return:
        """
