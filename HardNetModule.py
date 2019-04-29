import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


class HardNetModule(nn.Module):

    def __init__(self):
        # type: (HardNetModule)-> None
        """
        Construct a new HardNetModule object that will be responsible for holding the
        NN layers and providing access to optimizer variants.
        """
        super().__init__()
        self.model = self.__init_model()
        self.model.apply(self.__weight_init)

    def get_adam_optimizer(self, learning_rate, weight_decay):
        # type: (HardNetModule, float, float)->optim.Optimizer
        """
        Returns an ADAM optimizer with the given parameters
        :param learning_rate: The initial learning rate for the optimizer
        :param weight_decay: The weight decay coefficient to use with the optimizer
        :return: ADAM optimizer with the appropriate parameters
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
        return optimizer

    def get_sgd_optimizer(self, learning_rate, weight_decay, momentum=0.9, dampening=0.9):
        # type: (HardNetModule, float, float, float, float)->optim.Optimizer
        """
        Returns a standard SGD optimizer with the given parameters
        :param learning_rate: The initial learning rate for the optimizer
        :param weight_decay: The weight decay coefficient to use with the optimizer
        :param momentum: The momentum coefficient to use with SGD
        :param dampening: Dampening for momentum
        :return: SGD optimizer with the appropriate parameters
        """

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                              momentum=momentum, dampening=dampening,
                              weight_decay=weight_decay)

        return optimizer

    @staticmethod
    def __init_model():
        # type: ()-> nn.Sequential
        """
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
        # type: (nn.Module)-> None
        """
        Initializes the bias and weights of the neural network.
        For use with Module.apply(fn)

        :param module: The PyTorch Module to initialize
        """
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal(module.weight.data, gain=0.6)
            try:
                nn.init.constant(module.bias.data, 0.01)
            except Exception as e:
                print(type(e))
                # TODO: handle specific exception

    @staticmethod
    def input_norm(input_data):
        # type: (Tensor) -> Tensor
        """
        Normalizes each image in the input such that each image is 0 meaned
        and has a std deviation of 1
        :param input_data: A batch size x 32 x 32 x 1 tensor of inputs
        :return: A batch size x 32 x 32 x 1 tensor of normalized inputs
        """
        flat = input_data.view((input_data.size(0), -1))
        eps = 1e-10
        avg = torch.mean(flat, dim=(1,)).detach()
        std_dev = torch.add(torch.std(flat, dim=1).detach(), eps)

        avg_expanded = avg.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(input_data)
        std_dev_expanded = std_dev.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(input_data)

        # (input - avg_expanded) / std_dev_expanded
        return torch.div(
            torch.add(
                input_data,
                torch.mul(
                    avg_expanded,
                    -1
                )
            ),
            std_dev_expanded
        )

    @staticmethod
    def output_norm(output):
        # type: (Tensor) -> Tensor
        """
        Normalizes each image descriptor such that the norm of each
        descriptor is 1
        :param output: A Nx128 matrix of image descriptors
        :return: A Nx128 matrix of unit length image descriptors
        """
        eps = 1e-10
        norm = torch.sqrt(
            torch.add(
                torch.sum(
                    torch.mul(
                        output,
                        output
                    ),
                    dim=(1,)
                ),
                eps
            )
        ).unsqueeze(-1).expand_as(output)
        return torch.div(output, norm)

    def forward(self, input_data):
        # type: (Tensor) -> Tensor
        """
        Runs the input through the neural network layers

        :param input_data: A batch of 32x32x1 images
        :return: Batch size x 128 matrix of image descriptors
        """
        model_output = self.model(HardNetModule.input_norm(input_data))

        # reduce the size to batch size x 128
        reshaped_output = model_output.view(model_output.size(0), -1)

        return HardNetModule.output_norm(reshaped_output)
