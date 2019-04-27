from LossHardNet import LossHardNet


class LossHardNetTripletMargin(LossHardNet):
    def __init__(self, margin):
        super().__init__(margin)

    def impl(self, anchor, positive):
        """
        type: (Loss, Variable, Variable)-> Tensor
        TODO: finish commenting and implement
        :return:
        """
        pass
