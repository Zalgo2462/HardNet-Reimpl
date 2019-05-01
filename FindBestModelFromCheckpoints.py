import torch
import glob
from HardNet import HardNet
from HardNetModule import HardNetModule
import numpy as np
from PairPhotoTour import PairPhotoTour
from PairPhotoTourTestLoaderFactory import PairPhotoTourTestLoaderFactory

def __init_testing_loader_factories(testing_sets, batch_size, data_path):
    # type: (typing.List[str], int, str)->typing.List[AbstractDataloaderFactory]
    """
    Initialize and return the AbstractDataloaderFactories for the testing datasets specified.
    These factories are responsible for creating DataLoaders which provide testing samples
    to the network after training.

    :param testing_sets: names of the datasets to use for testing
    :param batch_size: number of anchor/positive pairs to include in a batch
    :param data_path: path to save data to or to read cached data from
    :return: The AbstractDataloaderFactories that will create Dataloaders which
             provide testing samples in pairs of batch size x 1 x 32 x 32 tensors
    """
    kwargs = {}
    # TODO: direct call to is_available might be replaced
    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': True}

    return [
        PairPhotoTourTestLoaderFactory(
            batch_size, data_root=data_path, name=testing_set, download=True, loader_kwargs=kwargs
        ) for testing_set in testing_sets
    ]

batch_size = 1024
data_path = 'data/sets/'
model_path = 'data/models/'

training_set_name = 'liberty'
testing_set_names = [name for name in PairPhotoTour.NAMES if name != training_set_name]

testing_loader_factories = __init_testing_loader_factories(testing_set_names, batch_size, data_path)

checkpoints = glob.glob("./data/models/*/*9.pth", recursive=True)

best_fpr_norm = 100000
best_fprs = []
best_checkpoint = ""

for checkpoint_path in checkpoints:
    print("Running {}".format(checkpoint_path))
    hnet = HardNet(HardNetModule(), "")
    hnet.load_checkpoint(checkpoint_path)
    fprs = np.array(hnet.get_fprs(testing_loader_factories))
    this_fpr_norm = np.sum(fprs ** 2) ** .5
    print("FPRs for {}: {}, {}".format(checkpoint_path, fprs[0], fprs[1]))
    if this_fpr_norm < best_fpr_norm:
        best_fpr_norm = this_fpr_norm
        best_fprs = fprs
        best_checkpoint = checkpoint_path

print("Best checkpoint: {}".format(best_checkpoint))

for i in range(len(testing_set_names)):
    name = testing_set_names[i]
    fpr = best_fprs[i]
    print("Best FPR for {}: {}".format(name, fpr))
    
    
