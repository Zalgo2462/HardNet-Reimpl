from torch.utils.data import DataLoader


class AbstractDataloaderFactory:
    def get_dataloader(self):
        # type: (AbstractDataloaderFactory) -> DataLoader
        """
        Returns a new DataLoader
        :return: a new DataLoader
        """
        pass
