import torch
import torch.nn as nn
from torch import Tensor


class HardNetModule(nn.Module):

    def __init__(self):
        # type: (HardNetModule)-> None
        """
        Construct a new HardNetModule object that will be responsible for holding the
        NN layers and providing access to optimizer variants.
        """
        super().__init__()
        self.__model = self.__init_model()
        self.__model.apply(self.__weight_init)

    def get_model(self):
        # type: (HardNetModule) -> nn.Module
        """
        Returns the nn layers used in the HardNetModule
        :return: the nn layers which make up the HardNetModule
        """
        return self.__model

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
        :param input_data: A batch size x 1 x 32 x 32 tensor of inputs
        :return: A batch size x 1 x 32 x 32 tensor of normalized inputs
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

        :param input_data: A batch of 1x32x32 images
        :return: Batch size x 128 matrix of image descriptors
        """

        model_output = self.__model(HardNetModule.input_norm(input_data))
        
        # reduce the size to batch size x 128
        reshaped_output = model_output.view(model_output.size(0), -1)

        return HardNetModule.output_norm(reshaped_output)
