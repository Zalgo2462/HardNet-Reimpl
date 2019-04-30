from typing import Any, Callable, Dict

from torch.utils.data import DataLoader
from torchvision.datasets import PhotoTour


class PairPhotoTour(PhotoTour):
    NAMES = ['liberty', 'notredame', 'yosemite']

    def __init__(self, batch_size, data_root, name, train, transform, download):
        # type: (PairPhotoTour, int, str, str, bool, Callable, bool)->None
        """
        Base class for PairPhotoTour training and testing variants. Specified the required
        interface inheritors are expected to implement, and provides the get_data_loader function shared by
        all implementors. Also passes named parameters to the PhotoTour

        :param batch_size: number of pairs to place in a batch
        :param data_root: path to the location to save downloaded samples/retrieve cached samples
        :param name: name of the dataset to use
        :param download: whether or not to download samples if they are not already present
        """
        super().__init__(root=data_root, name=name, train=train, transform=transform, download=download)
        self._batch_size = batch_size

    def __getitem__(self, index):
        # type: (int)->Any
        """
        Base implementation for the getitem function. Inheritors are expected to provide the index-th sample
        in the dataset.
        :param index: index of the sample to retrieve
        :return: sample from the data set, generally expected to be a tuple of Tensor objects
        """
        pass

    def __len__(self):
        # type: ()->int
        """
        Base implementation for the length function. Inheritors are expected to provide the number of samples
        in the dataset.
        :return: number of samples in the dataset
        """
        pass

    def get_data_loader(self, kwargs):
        # type: (Dict[str, Any])->DataLoader
        """
        Returns a DataLoader for the PairPhotoTour, allowing the dataset to be used as an input into the
        HardNet model.
        :return: DataLoader that can provide training or testing samples to the model
        """
        return DataLoader(self, batch_size=self._batch_size, shuffle=False, **kwargs)
