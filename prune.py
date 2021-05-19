import torch
from torch import nn
import torch.nn.utils.prune as prune

from models.simple_cnn import *
DEVICE = torch.device("cuda:1")


def prune(checkpoint_dir):
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_dir))
    model.eval()

