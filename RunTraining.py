import typing

import torch.cuda
from torch.utils.data import DataLoader

from HardNet import HardNet
from HardNetModule import HardNetModule
from PairPhotoTour import PairPhotoTour
from PairPhotoTourTest import PairPhotoTourTest
from PairPhotoTourTrain import PairPhotoTourTrain


def main():
    training_set_name = 'liberty'
    testing_set_names = [name for name in PairPhotoTour.NAMES if name != training_set_name]
    batch_size = 1024
    data_path = 'data/sets/'
    model_path = 'data/models/'

    training_loader = __init_training_loader(training_set_name, batch_size, data_path)
    testing_loaders = __init_testing_loader(testing_set_names, batch_size, data_path)
    hard_net = HardNet(HardNetModule(), model_path)


    pass


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
    :return: The DataLoader that will provide training pairs in pairs of batch size x 32 x 32 x 1 tensors
    """
    kwargs = {}
    # TODO: direct call to is_available might be replaced
    if torch.cuda.is_available():
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}

    return PairPhotoTourTrain(batch_size, data_root=data_path, download=True,
                              name=training_set).get_data_loader(kwargs)


def __init_testing_loader(testing_sets, batch_size, data_path, num_workers=0, pin_memory=True):
    # type: (typing.List[str], int, str, int, bool)->typing.List[DataLoader]
    """
    Initialize and return the DataLoaders for the testing datasets specified. These DataLoaders are responsible
    for providing testing samples to the network after training.

    :param testing_sets: names of the datasets to use for testing
    :param batch_size: number of anchor/positive pairs to include in a batch
    :param data_path: path to save data to or to read cached data from
    :param num_workers: subprocess count to use for data loading. If 0 data will be loaded in the main process.
    :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them.
    :return: The DataLoader that will provide testing samples in pairs of batch size x 32 x 32 x 1 tensors
    """
    kwargs = {}
    # TODO: direct call to is_available might be replaced
    if torch.cuda.is_available():
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}

    return [
        PairPhotoTourTest(
            batch_size, data_root=data_path, name=testing_set, download=True
        ).get_data_loader(kwargs)
        for testing_set in testing_sets
    ]


if __name__ == "__main__":
    main()
