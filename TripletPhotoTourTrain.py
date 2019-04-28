import random

import numpy as np
import torch

from TripletPhotoTour import TripletPhotoTour


class TripletPhotoTourTrain(TripletPhotoTour):
    def __init__(self, batch_size, *args, **kwargs):
        """
        TODO: finish commenting and implement
        :param batch_size:
        :param args:
        :param kwargs:
        """
        super().__init__(batch_size, *args, **kwargs)
        # TODO: give parameters
        self.pairs = self.__generate_pairs()

    def __generate_pairs(self, labels, num_pairs):
        """
        type: (List[str], int)->torch.LongTensor
        TODO: finish commenting and implement
        :param labels:
        :param num_pairs:
        """

        pairs = []
        indices = TripletPhotoTourTrain.__create_indices(labels.numpy())
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
        """
        type: (List[int])->Dict[int,List[int]]
        TODO: finish commenting and implement
        :param labels:
        :return:
        """
        indices = dict()
        for image_index, class_index in enumerate(labels):
            if class_index not in indices:
                indices[class_index] = []
            indices[class_index].append(image_index)

        return indices

    def __getitem__(self, index):
        """
        TODO: finish commenting and implement
        :param index:
        :return:
        """

        def transform_img(img):
            if self.transform is not None:
                # TODO: confirm if this returns an nd.array
                img = self.transform(img.numpy())
            return img

        t = self.pairs[index]
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
        """
        TODO: finish commenting and implement
        :return:
        """
        pass
