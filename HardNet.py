import torch.cuda
import torch.utils.data
from torch.utils.data import DataLoader

from HardNetModule import HardNetModule
from Logger import Logger
from TripletPhotoTourTest import TripletPhotoTourTest
from TripletPhotoTourTrain import TripletPhotoTourTrain


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

    @staticmethod
    def __init_training_loader(training_set, batch_size, data_path, num_workers=0, pin_memory=True):
        # type: (str, int, str, int, bool)->DataLoader
        """
        Initialize and return the DataLoader for the training dataset specified. This DataLoader is responsible
        for providing training pairs to the network.

        :param training_set: name of the dataset to use for training
        :param batch_size: number of anchor/positive pairs to include in a batch
        :param data_path: path to save data to or to read cached data from
        :param num_workers: subprocess count to use for data loading. If 0 data will be loaded in the main process.
        :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them.
        :return: The DataLoader that will provide training pairs
        """
        kwargs = {}
        # TODO: direct call to is_available might be replaced
        if torch.cuda.is_available():
            kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}

        return TripletPhotoTourTrain(batch_size, data_root=data_path, download=True,
                                     name=training_set).get_data_loader(kwargs)

    @staticmethod
    def __init_testing_loader(testing_set, batch_size, data_path, num_workers=0, pin_memory=True):
        # type: (str, int, str, int, bool)->DataLoader
        """
        Initalize and return the DataLoader for the testing dataset specified. This DataLoader is responsible
        for providing testing samples to the network after training.

        :param testing_set: name of the dataset to use for testing
        :param batch_size: number of anchor/positive pairs to include in a batch
        :param data_path: path to save data to or to read cached data from
        :param num_workers: subprocess count to use for data loading. If 0 data will be loaded in the main process.
        :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them.
        :return: The DataLoader that will provide testing samples
        """
        kwargs = {}
        # TODO: direct call to is_available might be replaced
        if torch.cuda.is_available():
            kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}

        return TripletPhotoTourTest(batch_size, data_root=data_path, name=testing_set, download=True).get_data_loader(
            kwargs)

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

    def train(self, logger):
        # type: (HardNet, Logger)->None
        """

        TODO: commenting and implementation
        :return:
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
