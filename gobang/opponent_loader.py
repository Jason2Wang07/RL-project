import torch.nn as nn
from typing import *
from utils import *
import numpy as np
from model_loader import get_model

board_size = 12
bound = 5


def get_opponent():
    return get_model("model.pth")


__all__ = ['get_opponent']
