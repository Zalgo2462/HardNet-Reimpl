import random
from typing import List, Dict

import PIL.Image
import numpy as np
import torch
import torchvision.transforms as transforms

from PairPhotoTour import PairPhotoTour


class PairPhotoTourTrain(PairPhotoTour):
    NUM_PAIRS = 5_000_000

    def __init__(self, batch_size, data_root, name, download):
        # type: (PairPhotoTourTrain, int, str, str, bool)->None
        """
        Create a PairPhotoTourTrain dataset that can provide a DataLoader for passing train data
        into the network.

        :param batch_size: number of training pairs to place in a batch
        :param data_root: path to the location to save downloaded samples/retrieve cached samples
        :param name: name of the dataset to use for training
        :param download: whether or not to download samples if they are not already present
        """
        tx = transforms.Compose([
            transforms.Lambda(lambda x: np.reshape(x, {64, 64, 1})),
            transforms.ToPILImage(),
            transforms.RandomRotation(5, PIL.Image.BILINEAR),
            transforms.RandomResizedCrop(32, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.Resize(32),
            transforms.ToTensor()])
        super().__init__(batch_size, data_root=data_root, name=name, train=True, transform=tx, download=download)
        self.__pairs = self.__generate_pairs(self.labels, self.NUM_PAIRS)

    def __generate_pairs(self, labels, num_pairs):
        # type: (torch.LongTensor, int)->torch.LongTensor
        """
        Create the pairs of anchors and positives that will be combined into batches for training.
        :param labels: image indices and classes that will be reorganized into a dictionary for ease of use
        :param num_pairs: number of pairs to create for training
        :return: A LongTensor containing all of the pairs generated, to be fed out by the DataLoader
        """

        pairs = []
        indices = PairPhotoTourTrain.__create_indices(labels.numpy())
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch

        already_idxs = set()

        # TODO: replace the tqdm call that was here previously
        for x in range(num_pairs):
            if len(already_idxs) >= self.__batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]))
                n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))
            pairs.append([indices[c1][n1], indices[c1][n2]])

        return torch.LongTensor(np.array(pairs))

    @staticmethod
    def __create_indices(labels):
        # type: (List[int])->Dict[int,List[int]]
        """
        For each class, create a list of the corresponding indices into the images set and store that list
        in a dictionary.
        :param labels: collection of index and class pairs to be sorted into lists of indices organized by class
        :return: dictionary of classes and their corresponding images
        """
        indices = dict()
        for image_index, class_index in enumerate(labels):
            if class_index not in indices:
                indices[class_index] = []
            indices[class_index].append(image_index)

        return indices

    def __getitem__(self, index):
        """
        return the index-th training pair consisting of the anchor and its positive.
        :param index: index of the pair to retrieve
        :return: a tuple containing the anchor and positive
        """

        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        t = self.__pairs[index]
        a, p = self.data[t[0]], self.data[t[1]]

        img_a = transform_img(a)
        img_p = transform_img(p)

        # always perform data augmentation
        do_flip = random.random() > 0.5
        do_rot = random.random() > 0.5
        if do_rot:
            # use torch to reorder the dimensions
            img_a = img_a.permute(0, 2, 1)
            img_p = img_p.permute(0, 2, 1)
        if do_flip:
            # TODO: MAKE SURE THIS FLIPS ON THE RIGHT AXIS
            img_a = torch.flip(img_a.numpy(), (2,))
            img_p = torch.flip(img_p.numpy(), (2,))
        return img_a, img_p

    def __len__(self):
        # type: ()->int
        """
        return the number of pairs in the dataset
        :return: the number of pairs in the dataset
        """
        return self.__pairs.size(0)
