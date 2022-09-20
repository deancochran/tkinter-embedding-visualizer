import math
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

device = 'cuda' if th.cuda.is_available() else 'cpu'
class Discriminator(nn.Module):
    def __init__(self, hidden_dim, out_dim, activation, num_layers=2, dropout=0.3, relu = 0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim =  out_dim
        self.relu=nn.LeakyReLU(relu)
        self.dropout=nn.Dropout(dropout)
        self.activation = activation

        layers =[]
        for layer in range(0,num_layers):
            if layer == 0 and num_layers>1:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim*2, bias=True))
                layers.append(self.relu)
                layers.append(self.dropout)
            elif num_layers<=1:
                layers.append(nn.Linear(self.hidden_dim, self.out_dim, bias=True))
            elif layer == (num_layers-1):
                layers.append(nn.Linear(self.hidden_dim*(layer*2), self.out_dim, bias=True))
            else:
                layers.append(nn.Linear(self.hidden_dim*(layer*2), self.hidden_dim*((layer+1)*2), bias=True))
                layers.append(self.relu)
                layers.append(self.dropout)
        self.layers=nn.Sequential(*layers)

    def forward(self, x):
        return self.activation(self.layers(x)).squeeze()
