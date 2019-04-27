import torch.nn as nn
import torch.optim as optim


class HardNetModule(nn.Module):

    def __init__(self):
        """
        type: (HardNetModule)-> None

        Construct a new HardNet model object
        """
        super().__init__()
        self.model = self.__init_model()
        self.model.apply(self.__weight_init)

    def get_adam_optimizer(self, learning_rate, weight_decay):
        """
        type: (HardNetModule, float, float)->optimizer.Optimizer

        TODO: finish commenting
        :return:
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
        return optimizer

    def get_sgd_optimizer(self, learning_rate, weight_decay, momentum=0.9, dampening=0.9):
        """
        type: (HardNetModule, float, float, float, float)->optimizer.Optimizer
        TODO: finish commenting
        :param learning_rate:
        :param weight_decay:
        :param momentum:
        :param dampening:
        :return:
        """

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                              momentum=momentum, dampening=dampening,
                              weight_decay=weight_decay)

        return optimizer

    @staticmethod
    def __init_model():
        """
        type: ()-> nn.Sequential

        Create the HardNet neural network layers
        """
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

    @staticmethod
    def __weight_init(module):
        """
        type: (nn.Module)-> None
        :param module: The PyTorch Module to initialize
        """
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal(module.weight.data, gain=0.6)
            try:
                nn.init.constant(module.bias.data, 0.01)
            except Exception as e:
                print(type(e))
                # TODO: handle specific exception

    def forward(self, *input):
        """
        TODO: Implement
        :param input:
        :return:
        """
        pass
