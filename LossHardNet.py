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

    def _get_negative(self, anchor, positive):
        """
        type: (LossHardNet, Variable, Variable)-> Variable
        TODO: finish commenting
        :param anchor:
        :param positive:
        :return:
        """
        pass

    def _distance_matrix_vector(self, positive, anchor):
        """
        type: (LossHardNet, Variable, Variable)-> Variable
        TODO: finish commenting
        TODO: get a better name
        :param positive:
        :param anchor:
        :return:
        """
        pass

    @staticmethod
    def _filter_distance_matrix(distances):
        """
        type: (Variable)->Variable
        TODO: commenting and implementation
        :param distances:
        :return:
        """
        pass
