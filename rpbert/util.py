import torch
import numpy as np


def to_tensor(array):
    return torch.from_numpy(np.array(array)).float()


def to_variable(tensor, requires_grad=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, requires_grad=requires_grad)
