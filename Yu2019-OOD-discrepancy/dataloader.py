from PIL import Image
import os
import random
import os.path
import numpy as np
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data

#dataloader
import warnings
warnings.filterwarnings("ignore")

# Python
import random
import argparse

# Torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# Torchvison
from torchvision.utils import make_grid
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, MNIST

#Custom
#from config import *

def dataset_loader(sup_train='',sup_val='',sup_test='',unsup_train='',unsup_val='',unsup_test='',BATCH=64):
    indices = list(range(10000))
    random.Random(4).shuffle(indices)
    dataloaders={}
    if sup_train:
        train_loader = DataLoader(sup_train, batch_size=BATCH//2,
                                  shuffle=True, pin_memory=True, 
                                  drop_last=True, num_workers=2)
        dataloaders['sup_train']=(train_loader)
    if sup_val:
        val_loader = DataLoader(sup_val, batch_size=BATCH,
                                shuffle=True,
                                pin_memory=True, num_workers=2)
        dataloaders['sup_val']=val_loader
    if sup_test:
        test_loader = DataLoader(sup_test, batch_size=BATCH,
                                 shuffle=True, 
                                 pin_memory=True, num_workers=2)
        dataloaders['sup_test']=test_loader
    if unsup_train:
        unsup_train_loader = DataLoader(unsup_train, batch_size=BATCH//2,
                                        shuffle=True, pin_memory=True, 
                                        drop_last=True, num_workers=2)
        dataloaders['unsup_train']=list(unsup_train_loader)
    if unsup_val:
        unsup_val_loader = DataLoader(unsup_val, batch_size=BATCH,
                                      shuffle=False, pin_memory=True, 
                                      num_workers=2)
        dataloaders['unsup_val']=(unsup_val_loader)
    if unsup_test:
        unsup_test_loader = DataLoader(unsup_test, batch_size=BATCH,
                                      shuffle=False, pin_memory=True, 
                                      num_workers=2)
        dataloaders['unsup_test']=unsup_test_loader
            
    
    return dataloaders


    
            




