from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms

from PairPhotoTour import PairPhotoTour


class PairPhotoTourTest(PairPhotoTour):
    def __init__(self, batch_size, data_root, name, download):
        # type: (PairPhotoTourTest, int, str, str, bool)->None
        """
        Create a PairPhotoTourTest dataset that can provide a DataLoader for passing test data
        into the network.

        :param batch_size: number of test samples to place in a batch
        :param data_root: path to the location to save downloaded samples/retrieve cached samples
        :param name: name of the dataset to use for testing
        :param download: whether or not to download samples if they are not already present
        """
        tx = transforms.Compose([
            transforms.Lambda(lambda x: np.reshape(x, {64, 64, 1})),
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor()  # has the side effect of reordering dimensions to CxHxW
        ])
        super().__init__(batch_size, data_root=data_root, name=name, train=False, transform=tx,
                         download=download)

    def __getitem__(self, index):
        # type: (int)->Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
        """
        return the index-th testing sample consisting of the sample, its comparison, and its label.
        :param index: index of the samples to retrieve
        :return: a tuple containing the sample, comparison, and label (which is a 1 if they are a match or a 0 if not)
        """

        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        m = self.matches[index]
        img1 = transform_img(self.data[m[0]])
        img2 = transform_img(self.data[m[1]])

        return img1, img2, m[2]

    def __len__(self):
        # type: ()->int
        """
        return the number of samples in the dataset
        :return: the number of samples in the dataset
        """
        return self.matches.size(0)
