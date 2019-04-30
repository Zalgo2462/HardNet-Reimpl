import typing

import torch.cuda

from AbstractDataloaderFactory import AbstractDataloaderFactory
from HardNet import HardNet
from HardNetModule import HardNetModule
from PairPhotoTour import PairPhotoTour
from PairPhotoTourTestLoaderFactory import PairPhotoTourTestLoaderFactory
from PairPhotoTourTrainLoaderFactory import PairPhotoTourTrainLoaderFactory


def main():
    training_set_name = 'liberty'
    testing_set_names = [name for name in PairPhotoTour.NAMES if name != training_set_name]
    batch_size = 1024
    data_path = 'data/sets/'
    model_path = 'data/models/'

    training_loader_factory = __init_training_loader_factory(training_set_name, batch_size, data_path)
    testing_loader_factories = __init_testing_loader_factories(testing_set_names, batch_size, data_path)
    hard_net = HardNet(HardNetModule(), model_path)

    pass


def __init_training_loader_factory(training_set, batch_size, data_path):
    # type: (str, int, str)->AbstractDataloaderFactory
    """
    Initialize and return the AbstractDataloaderFactory for the training dataset specified.
    This factory is responsible for creating DataLoaders which provide training pairs to the network.

    :param training_set: name of the dataset to use for training
    :param batch_size: number of anchor/positive pairs to include in a batch
    :param data_path: path to save data to or to read cached data from
    :return: The AbstractDataloaderFactory that will create Dataloaders which
             provide training pairs in pairs of batch size x 1 x 32 x 32 tensors
    """
    kwargs = {}
    # TODO: direct call to is_available might be replaced
    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': True}

    return PairPhotoTourTrainLoaderFactory(batch_size, data_root=data_path, download=True,
                                           name=training_set, loader_kwargs=kwargs)


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
