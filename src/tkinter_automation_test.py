import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch_geometric.nn import MetaPath2Vec
import torch_geometric.transforms as T

from dataloader import load_ML100K
from dataset import CustomDataset
from ml100k_pyg_loader import ML100k
from model import Discriminator, Encoder
from utils import activation, criterion, optimizer, roc_auc

device = 'cuda' if th.cuda.is_available() else 'cpu'

dataset = ML100k('../data/ml-100k')
hg = dataset[0]
hg = T.ToUndirected()(hg)

metapatorch = [
    ('user', 'rates', 'movie'),
    ('movie', 'rev_rates', 'user'),
]

metapath2vec = MetaPath2Vec(hg.edge_index_dict, embedding_dim=128,
                     metapath=metapatorch, walk_length=50, context_size=7,
                     walks_per_node=5, num_negative_samples=5,
                     sparse=True).to(device)

loader = metapath2vec.loader(batch_size=128, shuffle=True, num_workers=6)
optimizer = torch.optim.SparseAdam(list(metapath2vec.parameters()), lr=0.01)
epochs = 5
metapath2vec.train()
for epoch in range(1, epochs + 1):
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = metapath2vec.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        print('\r',f'Epoch: {epoch} of {epochs+1}, Step: {i + 1:.3f}/{len(loader)}, 'f'Loss: {loss:.3f}', end=' ')

embedding_dict = {}
for node_type in metapath2vec.num_nodes_dict:
    # get embedding of node witorch specific type
    embedding_dict[node_type] = metapath2vec(node_type).detach().cpu()
    print(node_type, embedding_dict[node_type].shape)

from sklearn.manifold import TSNE
X = TSNE(n_components=2, init='random', perplexity=3).fit_transform(embedding_dict['user'])
y_true = hg['user'].gender
