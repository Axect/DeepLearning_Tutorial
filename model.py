import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hparams):
        super(MLP, self).__init__()
        self.hparams = hparams

        nodes = hparams['nodes']
        layers = hparams['layers']

        net = [nn.Linear(1, nodes), nn.ReLU()]
        for _ in range(layers - 1):
            net.extend([nn.Linear(nodes, nodes), nn.ReLU()])
        net.append(nn.Linear(nodes, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
