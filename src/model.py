import math
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

device = 'cuda' if th.cuda.is_available() else 'cpu'
class Discriminator(nn.Module):
    def __init__(self, hidden_dim, out_dim, activation, num_layers=2, dropout=0.3, relu = 0.2):
        """
        Discriminator is a generic pytorch machine learning model that can offer a variably large amount of layers to preform deep learning
        - choice of the layers is not available, all are set to be Linear

        Args:
            hidden_dim (int): the size of the input used to train the discriminator
            out_dim (int): The size of the output dimension
            activation (torch.nn): the torch.nn activation method that will be used to make forward predictions
            num_layers (int, optional): the number of Linear layers the model will have. Defaults to 2.
            dropout (float, optional): the dropout rate. Defaults to 0.3.
            relu (float, optional): the LeakyRelu rate. Defaults to 0.2.
        """
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
