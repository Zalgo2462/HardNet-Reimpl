import typing

import numpy as np
import torch.cuda
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from AbstractDataloaderFactory import AbstractDataloaderFactory
from AbstractOptimizerFactory import AbstractOptimizerFactory
from HardNetModule import HardNetModule
from Logger import Logger
from LossHardNet import LossHardNet
from StatsUtils import false_positive_rate_at_95_recall


class HardNet:
    def __init__(self, hard_net_module, model_path):
        # type: (HardNet, HardNetModule, str)->None
        """
        Initialize a new HardNet controller object

        :param hard_net_module: The HardNet NN layers
        :param model_path: The path to create and load checkpoints to/from
        """
        self.__module = hard_net_module
        self.__model_path = model_path
        self.__current_epoch = 0

        if torch.cuda.is_available():
            self.__module = self.__module.cuda()

    def save_checkpoint(self):
        # type: (HardNet)->None
        """

        TODO: commenting and implementation
        :return:
        """
        pass

    def load_checkpoint(self, checkpoint_path):
        # type: (HardNet, str)->None
        """

        TODO: commenting and implementation
        :param checkpoint_path:
        :return:
        """
        pass

    def train(
            self,  # type: HardNet
            training_loader_factory,  # type: AbstractDataloaderFactory
            testing_loader_factories,  # type: typing.List[AbstractDataloaderFactory]
            optimizer_factory,  # type: AbstractOptimizerFactory
            hardnet_loss,  # type: LossHardNet
            end_epoch,  # type:  int
            logger  # type: Logger
    ):
        # type: (...)->None
        """
        :param training_loader_factory: A Dataloader factory which creates Dataloaders which return training pairs
               in pairs of batch size x 32 x 32 x 1 tensors
        :param testing_loader_factories: A list of Dataloader factories which create Dataloaders which return test pairs
               in the same format as training_loader
        :param optimizer_factory: creates the optimizer to use while training
        :param hardnet_loss: loss object to score with
        :param end_epoch: epoch to end training at
        :param logger: A logger to record training progress
        """

        training_loader = training_loader_factory.get_dataloader()
        testing_loaders = [factory.get_dataloader() for factory in testing_loader_factories]

        optimizer = optimizer_factory.create_optimizer(self.__module.get_model().parameters())

        num_batches_per_epoch = len(training_loader)
        total_num_batches = num_batches_per_epoch * end_epoch

        def lr_schedule_fun(batch_idx):
            return 1.0 - batch_idx / total_num_batches

        lr_scheduler = LambdaLR(optimizer, lr_schedule_fun, self.__current_epoch * num_batches_per_epoch)

        # TODO: resume functionality omitted here, need to implement using the load function

        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, end_epoch):
            self.__train_epoch(epoch, training_loader, optimizer, hardnet_loss, lr_scheduler, logger)

            for testing_loader in testing_loaders:
                self.__test_epoch(epoch, testing_loader, logger)

            training_loader = training_loader_factory.get_dataloader()

    def __train_epoch(self, epoch, training_loader, optimizer, hardnet_loss, lr_scheduler, logger):
        # type: (HardNet, int, DataLoader, Optimizer, LossHardNet,LambdaLR, Logger)->None
        """
        :param epoch: current epoch being trained
        :param training_loader: A dataloader which returns training pairs in pairs of batch size x 32 x 32 x 1 tensors
        :param optimizer: optimizer to apply gradients with
        :param hardnet_loss: loss object to score with
        :param lr_scheduler: scheduler to manage reducing the learning_rate over steps
        :param logger: A logger to record training progress
        """
        self.__module.train()
        # TODO: find replacement for progress bar that was here
        for batch_index, (batch_anchors, batch_positives) in enumerate(training_loader):
            # TODO: replace direct check for cuda.is_available()
            if torch.cuda.is_available():
                batch_anchors, batch_positives = batch_anchors.cuda(), batch_positives.cuda()
                # TODO: determine if following line is needed
                # batch_anchors, batch_positives = Variable(data_a), Variable(data_p)
            out_anchors = self.__module(batch_anchors)
            out_positives = self.__module(batch_positives)

            loss_value = hardnet_loss.impl(out_anchors, out_positives)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            lr_scheduler.step()

            # if logger is not None:
            #     # log the loss if the logger exists
            #     logger.log('loss', loss_value,)
            #
            # # TODO: (try to) create the model dir if it doesn't exist
            # try:
            #     # TODO: specify suffix
            #     os.stat('{0}{1}'.format(self.__model_path, suffix))
            # # TODO: specify more specific exception
            # except Exception as e:
            #     # TODO: specify suffix
            #     os.makedirs('{0}{1}'.format(self.__model_path, suffix))
            #
            # # save the checkpoint for this epoch
            # self.save_checkpoint()

    def __test_epoch(self, epoch, testing_loader, logger):
        # type: (HardNet, int, DataLoader, Logger)->None
        """

        TODO: commenting and implementation
        :return:
        """
        self.__module.eval()

        distances, labels = [], []

        # TODO: find replacement for progress bar that was here
        for batch_index, (batch_anchors, batch_positives, batch_labels) in enumerate(testing_loader):
            # TODO: replace direct check for cuda.is_available()
            if torch.cuda.is_available():
                batch_anchors, batch_positives = batch_anchors.cuda(), batch_positives.cuda()
                # TODO: determine if following line is needed
                # batch_anchors, batch_positives = Variable(data_a), Variable(data_p)
            out_anchors = self.__module(batch_anchors)  # type: Tensor
            out_positives = self.__module(batch_positives)  # type: Tensor

            # dists = torch.sqrt(torch.sum((out_anchors - out_positives) ** 2, 1))
            dists = torch.sqrt(
                torch.sum(
                    torch.pow(
                        torch.add(
                            out_anchors,
                            torch.mul(
                                out_positives,
                                -1.0
                            )
                        ),
                        2.0
                    ),
                    (1,)
                )
            )

            distances.append(dists.cpu().numpy().reshape(1, -1))

            labels.append(batch_labels.cpu().numpy().reshape(1, -1))

            # TODO: LOGGING

        labels_combined = np.hstack(labels)
        distances_combined = np.hstack(distances)

        false_positive_rate = false_positive_rate_at_95_recall(labels_combined, distances_combined)

        # TODO: LOGGING
