import random
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import datasets
from tqdm import tqdm
from copy import deepcopy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        if type(t) is np.ndarray:
            t = torch.from_numpy(t)
        t = t.cuda()
    return Variable(t, **kwargs)





class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):
        self.model = model
        self.dataset = dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        # Use a deepcopy of the model to ensure the original model is not modified
        model_copy = deepcopy(self.model)
        model_copy.eval()

        for input in self.dataset:
            model_copy.zero_grad()
            input = variable(input)
            input = input[None, :]
            output = model_copy(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward(retain_graph=True)
            for n, p in model_copy.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
