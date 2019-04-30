from torch.utils.data import DataLoader

from AbstractDataloaderFactory import AbstractDataloaderFactory
from PairPhotoTourTest import PairPhotoTourTest


class PairPhotoTourTestLoaderFactory(AbstractDataloaderFactory):
    def __init__(self, batch_size, data_root, name, download, loader_kwargs):
        """
        :param batch_size: number of Testing pairs to place in a batch
        :param data_root: path to the location to save downloaded samples/retrieve cached samples
        :param name: name of the dataset to use for Testing
        :param download: whether or not to download samples if they are not already present
        :param loader_kwargs: kwargs to send to the DataLoader constructor (num_workers, pin_memory, etc.)
        """
        self.__batch_size = batch_size
        self.__data_root = data_root
        self.__name = name
        self.__download = download
        self.__loader_kwargs = loader_kwargs

    def get_dataloader(self):
        # type: (PairPhotoTourTestLoaderFactory) -> DataLoader
        """
        Returns a new DataLoader for PairPhotoTourTest
        :return: a new DataLoader for PairPhotoTourTest
        """

        return PairPhotoTourTest(
            self.__batch_size,
            self.__data_root,
            self.__name,
            self.__download,
        ).get_data_loader(
            self.__loader_kwargs
        )
