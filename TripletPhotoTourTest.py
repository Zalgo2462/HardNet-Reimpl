from TripletPhotoTour import TripletPhotoTour


class TripletPhotoTourTest(TripletPhotoTour):
    def __init__(self, batch_size, *args, **kwargs):
        """
        TODO: finish commenting and implement
        :param batch_size:
        :param args:
        :param kwargs:
        """
        super().__init__(batch_size, *args, **kwargs)

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
