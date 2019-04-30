import os
import typing

import torch.cuda

from AbstractDataloaderFactory import AbstractDataloaderFactory
from HardNet import HardNet
from HardNetModule import HardNetModule
from Logger import Logger
from LoggerConsole import LoggerConsole
from LossHardNetTripletMargin import LossHardNetTripletMargin
from PairPhotoTour import PairPhotoTour
from PairPhotoTourTestLoaderFactory import PairPhotoTourTestLoaderFactory
from PairPhotoTourTrainLoaderFactory import PairPhotoTourTrainLoaderFactory
from SGDOptimizerFactory import SGDOptimizerFactory


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")

    training_set_name = 'liberty'
    testing_set_names = [name for name in PairPhotoTour.NAMES if name != training_set_name]
    batch_size = 1024
    data_path = 'data/sets/'
    model_path = 'data/models/'

    logger_console = LoggerConsole()
    training_loader_factory = __init_training_loader_factory(training_set_name,
                                                             batch_size, data_path, logger_console, 100)
    testing_loader_factories = __init_testing_loader_factories(testing_set_names, batch_size, data_path)
    optimizer_factory = SGDOptimizerFactory(1, 0.0001, 0.9, 0.9)
    loss_triplet_margin = LossHardNetTripletMargin(1)
    hard_net = HardNet(HardNetModule(), model_path)

    experiment_tag = os.environ["HOSTNAME"] + "_run_1"
    hard_net.train(training_loader_factory, testing_loader_factories, optimizer_factory, loss_triplet_margin, 10,
                   experiment_tag, logger_console, 100)

    return


def __init_training_loader_factory(training_set, batch_size, data_path, logger, log_cycle):
    # type: (str, int, str, Logger, int)->AbstractDataloaderFactory
    """
    Initialize and return the AbstractDataloaderFactory for the training dataset specified.
    This factory is responsible for creating DataLoaders which provide training pairs to the network.

    :param training_set: name of the dataset to use for training
    :param batch_size: number of anchor/positive pairs to include in a batch
    :param data_path: path to save data to or to read cached data from
    :param logger: logging object to record false positive rates and other information
    :param log_cycle:number of batches between logging events during training
    :return: The AbstractDataloaderFactory that will create Dataloaders which
             provide training pairs in pairs of batch size x 1 x 32 x 32 tensors
    """
    kwargs = {}
    # TODO: direct call to is_available might be replaced
    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': True}

    return PairPhotoTourTrainLoaderFactory(batch_size, data_root=data_path, download=True,
                                           name=training_set, loader_kwargs=kwargs, logger=logger, log_cycle=log_cycle)


def __init_testing_loader_factories(testing_sets, batch_size, data_path):
    # type: (typing.List[str], int, str)->typing.List[AbstractDataloaderFactory]
    """
    Initialize and return the AbstractDataloaderFactories for the testing datasets specified.
    These factories are responsible for creating DataLoaders which provide testing samples
    to the network after training.

    :param testing_sets: names of the datasets to use for testing
    :param batch_size: number of anchor/positive pairs to include in a batch
    :param data_path: path to save data to or to read cached data from
    :return: The AbstractDataloaderFactories that will create Dataloaders which
             provide testing samples in pairs of batch size x 1 x 32 x 32 tensors
    """
    kwargs = {}
    # TODO: direct call to is_available might be replaced
    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': True}

    return [
        PairPhotoTourTestLoaderFactory(
            batch_size, data_root=data_path, name=testing_set, download=True, loader_kwargs=kwargs
        ) for testing_set in testing_sets
    ]


if __name__ == "__main__":
    main()
