from __future__ import print_function

import numpy as np

from models.unet import tinyUnet
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

def net_model(config: dict):
    if config['network'] == 'tinyUnet':
        net = tinyUnet()          
    else:
        assert False
    return net