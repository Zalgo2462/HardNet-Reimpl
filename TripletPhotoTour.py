from torchvision.datasets import PhotoTour


class TripletPhotoTour(PhotoTour):
    def __init__(self, batch_size, *args, **kwargs):
        """
        TODO: finish commenting and implement
        :param batch_size:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.__batch_size = batch_size

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

    def get_data_loader(self):
        """
        TODO: finish commenting
        :return:
        """
        pass
