from TripletPhotoTour import TripletPhotoTour


class TripletPhotoTourTrain(TripletPhotoTour):
    def __init__(self, *args, **kwargs):
        """
        TODO: finish commenting and implement
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def __generate_triplets(labels, num_triplets):
        """
        type: (List[str], int)->torch.LongTensor
        TODO: finish commenting and implement
        :param labels:
        :param num_triplets:
        """
        pass

    def __create_indices(self, labels):
        """
        type: (List[str])->Dict[int,List[int]]
        TODO: finish commenting and implement
        :param labels:
        :return:
        """
        pass

    def __getitem__(self, item):
        """
        TODO: finish commenting and implement
        :param item:
        :return:
        """
        pass

    def __len__(self):
        """
        TODO: finish commenting and implement
        :return:
        """
        pass
