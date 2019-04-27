import torch.cuda
import torch.utils.data

from HardNetModule import HardNetModule


class HardNet:
    def __init__(self):
        """
        type: (HardNet)->None
        """
        self.module = HardNetModule()

    @staticmethod
    def __init_training_loader(training_set, batch_size, data_path, num_workers=0, pin_memory=True):
        """
        type: (str, int, str, int, bool)->torch.utils.data.DataLoader
        TODO: commenting and implementation
        :return:
        """
        kwargs = {}
        # TODO: direct call to is_available might be replaced
        if torch.cuda.is_available():
            kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}

        train_loader = torch.utils.data.DataLoader()

        return train_loader

    @staticmethod
    def __init_testing_loader(data_set_names, batch_size, data_path, num_workers=0, pin_memory=True):
        """
        type: (List[str], int, str, int, bool)->torch.utils.data.DataLoader
        TODO: commenting and implementation
        :return:
        """
        kwargs = {}
        # TODO: direct call to is_available might be replaced
        if torch.cuda.is_available():
            kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}

        test_loader = torch.utils.data.DataLoader()

        return test_loader

    def train(self, logger):
        """
        type: (HardNet, Logger)->None
        TODO: commenting and implementation
        :return:
        """
        pass

    def __train_epoch(self, epoch):
        """
        type: (HardNet, int)->None
        TODO: commenting and implementation
        :return:
        """
        pass

    def __test_epoch(self, epoch):
        """
        type: (HardNet, int)->None
        TODO: commenting and implementation
        :return:
        """
        pass
