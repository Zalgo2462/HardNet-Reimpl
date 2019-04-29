import typing

from torch.utils.data import DataLoader

from HardNetModule import HardNetModule
from Logger import Logger


class HardNet:
    def __init__(self, hard_net_module, model_path):
        # type: (HardNet, HardNetModule, str)->None
        """
        Initialize a new HardNet controller object

        :param hard_net_module: The HardNet NN layers
        :param model_path: The path to create and load checkpoints to/from
        """
        self.__module = hard_net_module
        self.__model_path = model_path
        self.__total_epochs = -1
        self.__current_epoch = -1

    def save_checkpoint(self):
        # type: (HardNet)->None
        """

        TODO: commenting and implementation
        :return:
        """
        pass

    def load_checkpoint(self, checkpoint_path):
        # type: (HardNet, str)->None
        """

        TODO: commenting and implementation
        :param checkpoint:
        :return:
        """
        pass

    def train(self, training_loader, testing_loaders, logger):
        # type: (HardNet, DataLoader, typing.List[DataLoader], Logger)->None
        """
        :param training_loader: A dataloader which returns training pairs in pairs of batch size x 32 x 32 x 1 tensors
        :param testing_loaders: A list of dataloaders which return test pairs in the same format as training_loader
        :param logger: A logger to record training progress
        """
        pass

    def __train_epoch(self, epoch):
        # type: (HardNet, int)->None
        """

        TODO: commenting and implementation
        :return:
        """
        pass

    def __test_epoch(self, epoch):
        # type: (HardNet, int)->None
        """

        TODO: commenting and implementation
        :return:
        """
        pass
