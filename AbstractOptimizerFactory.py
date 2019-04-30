from typing import Iterable

from torch import Tensor
from torch.optim import Optimizer


class AbstractOptimizerFactory:

    def create_optimizer(self, module_parameters):
        # type: (AbstractOptimizerFactory, Iterable[Tensor])->Optimizer
        """
        Base implementation of CreateOptimizer, returns an Optimizer initialized with the model parameters in
        model_parameters.
        :param module_parameters: retrieved from module.parameters()
        :return: Optimizer initialized with the module parameters
        """
        pass
