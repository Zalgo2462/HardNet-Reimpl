import os
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

    def save_checkpoint(self, experiment_tag):
        # type: (HardNet, str)->None
        """
        Save a checkpoint of the module so that training can resume from the current state.

        :param experiment_tag: added to checkpoint directory as a suffix
        """
        experiment_dir = '{}{}'.format(self.__model_path, experiment_tag)

        os.makedirs(experiment_dir, exist_ok=True)

        torch.save({'epoch': self.__current_epoch + 1, 'state_dict': self.__module.state_dict()},
                   '{}/checkpoint_{}.pth'.format(experiment_dir, self.__current_epoch))

    def load_checkpoint(self, checkpoint_path):
        # type: (HardNet, str)->None
        """
        Attempts to load the starting epoch and state dictionary from the provided file path. If the load
        fails, the proper exception will be raised. If the load succeeds the HardNetModule will be updated
        to match the state dictionary retrieved and the current epoch will equal the one that was saved.
        :param checkpoint_path: path to find the checkpoint file at
        """
        checkpoint = torch.load(checkpoint_path)
        self.__current_epoch = checkpoint['epoch']
        self.__module.load_state_dict(checkpoint['state_dict'])

    def create_descriptors(self, input):
        """
        #TODO: comment!
        :param input:
        :return:
        """
        with torch.no_grad():
            if torch.cuda.is_available():
                input = input.cuda()

            return self.__module(input).cpu()

    def train(
            self,  # type: HardNet
            training_loader_factory,  # type: AbstractDataloaderFactory
            testing_loader_factories,  # type: typing.List[AbstractDataloaderFactory]
            optimizer_factory,  # type: AbstractOptimizerFactory
            hardnet_loss,  # type: LossHardNet
            end_epoch,  # type:  int
            experiment_tag,  # type: str
            logger,  # type: Logger
            log_cycle,  # type: int
    ):
        # type: (...)->None
        """
        :param training_loader_factory: A Dataloader factory which creates Dataloaders which return training pairs
               in pairs of batch size x 1 x 32 x 32 tensors
        :param testing_loader_factories: A list of Dataloader factories which create Dataloaders which return test pairs
               in the same format as training_loader
        :param optimizer_factory: creates the optimizer to use while training
        :param hardnet_loss: loss object to score with
        :param end_epoch: epoch to end training at
        :param experiment_tag: string appended to the checkpoint directory name
        :param logger: A logger to record training progress
        :param log_cycle: number of batches between logging events during training
        """

        training_loader = training_loader_factory.get_dataloader()
        testing_loaders = [factory.get_dataloader() for factory in testing_loader_factories]

        optimizer = optimizer_factory.create_optimizer(self.__module.get_model().parameters())

        num_batches_per_epoch = len(training_loader)
        total_num_batches = num_batches_per_epoch * end_epoch

        def lr_schedule_fun(batch_idx):
            return 1.0 - batch_idx / total_num_batches

        lr_scheduler = LambdaLR(optimizer, lr_schedule_fun, -1)

        # set the scheduler to the current epoch
        for i in range(-1, self.__current_epoch * num_batches_per_epoch):
            lr_scheduler.step()

        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, end_epoch):
            self.__train_epoch(epoch, training_loader, optimizer, hardnet_loss, lr_scheduler, logger, log_cycle)

            # save the checkpoint for this epoch
            self.save_checkpoint(experiment_tag)

            for testing_loader in testing_loaders:
                self.__test_epoch(epoch, testing_loader, logger, log_cycle)

            training_loader = training_loader_factory.get_dataloader()

            self.__current_epoch += 1

    def __train_epoch(self, epoch, training_loader, optimizer, hardnet_loss, lr_scheduler, logger, log_cycle):
        # type: (HardNet, int, DataLoader, Optimizer, LossHardNet,LambdaLR, Logger, int)->None
        """
        :param epoch: current epoch being trained
        :param training_loader: A dataloader which returns training pairs in pairs of batch size x 1 x 32 x 32 tensors
        :param optimizer: optimizer to apply gradients with
        :param hardnet_loss: loss object to score with
        :param lr_scheduler: scheduler to manage reducing the learning_rate over steps
        :param logger: A logger to record training progress
        :param log_cycle: number of batches between logging events during training
        """
        self.__module.train()
        loss_value = None  # type: typing.Union[Tensor, None]
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
            
            if logger is not None and batch_index % log_cycle == 0:
                # log the loss if the logger exists
                logger.log({'epoch': epoch, 'loss_value': loss_value, 'batch_index': batch_index, 'mode': 'train'})

        logger.log({'epoch': epoch, 'loss_value': loss_value, 'mode': 'train', 'reportable': True})

    def get_fprs(self, testing_loader_factories):
        fprs = []
        for factory in testing_loader_factories:
            testing_loader = factory.get_dataloader()

            self.__module.eval()
            distances, labels = [], []

            with torch.no_grad():
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

                    distances.append(dists.data.cpu().numpy().flatten())

                    labels.append(batch_labels.data.cpu().numpy().flatten())

            labels_combined = np.hstack(labels)
            distances_combined = np.hstack(distances)

            false_positive_rate = false_positive_rate_at_95_recall(labels_combined, distances_combined)
           
            fprs.append(false_positive_rate)
        return fprs 
        

    def __test_epoch(self, epoch, testing_loader, logger, log_cycle):
        # type: (HardNet, int, DataLoader, Logger, int)->None
        """
        Run a single epoch of tests and report false positive rate through the provided logger.
        :param epoch: current epoch the net is running in
        :param testing_loader: DataLoader which provides testing samples in pairs of batch size x 1 x 32 x 32 tensors
        :param logger: logging object to record false positive rates and other information
        :param log_cycle:number of batches between logging events during training
        """
        self.__module.eval()
        distances, labels = [], []

        with torch.no_grad():
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

                    distances.append(dists.data.cpu().numpy().flatten())

                    labels.append(batch_labels.data.cpu().numpy().flatten())

                    if logger is not None and batch_index % log_cycle == 0:
                        # log the loss if the logger exists
                        logger.log({'epoch': epoch, 'batch_index': batch_index, 'mode': 'test'})

        labels_combined = np.hstack(labels)
        distances_combined = np.hstack(distances)

        false_positive_rate = false_positive_rate_at_95_recall(labels_combined, distances_combined)

        logger.log({'epoch': epoch, 'FPR_at_95': false_positive_rate, 'mode': 'test', 'reportable': True})
