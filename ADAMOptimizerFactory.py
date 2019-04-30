from typing import Iterable

from torch import Tensor
from torch.optim import Optimizer, SGD

from AbstractOptimizerFactory import AbstractOptimizerFactory


class ADAMOptimizerFactory(AbstractOptimizerFactory):

    def __init__(self, learning_rate, weight_decay):
        # type: (ADAMOptimizerFactory, float, float)->None
        """
        Initialize the factory with the learning rate,
        and weight_decay to apply when create_optimizer is called.
        :param learning_rate: The initial learning rate for the optimizer
        :param weight_decay: The weight decay coefficient to use with the optimizer
        """
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay

    def create_optimizer(self, module_parameters):
        # type: (ADAMOptimizerFactory, Iterable[Tensor])->Optimizer
        """
        returns an Optimizer initialized with the model parameters in
        model_parameters.
        :param module_parameters: retrieved from module.parameters()
        :return: Optimizer initialized with the module parameters
        """
        return SGD(module_parameters, lr=self.__learning_rate, weight_decay=self.__weight_decay)
