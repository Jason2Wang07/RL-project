import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch
from submission import GobangModel

board_size = 12
bound = 5


def get_model(model_file="model.pth"):
	model = GobangModel(board_size=board_size, bound=bound)
	model.load_state_dict(torch.load(model_file))
	model.to(device)
	return model


__all__ = ['get_model']
