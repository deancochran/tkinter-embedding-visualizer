import math
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

''' Some Helpful Globals '''
device = 'cuda' if th.cuda.is_available() else 'cpu'

def apply_filters_gcmc(emb, filter_list):
    filter_emb = 0
    for f in filter_list:
        if f is not None:
            filter_emb = filter_emb + f(emb)
    return filter_emb

class SharedBilinearDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    """

    def __init__(self, num_relations, num_weights, embed_dim):
        super(SharedBilinearDecoder, self).__init__()
        self.rel_embeds = nn.Embedding(num_weights, embed_dim*embed_dim)
        self.weight_scalars = nn.Parameter(th.Tensor(num_weights,num_relations))
        stdv = 1. / math.sqrt(self.weight_scalars.size(1))
        self.weight_scalars.data.uniform_(-stdv, stdv)
        self.embed_dim = embed_dim
        self.num_weights = num_weights
        self.num_relations = num_relations
        self.nll = nn.NLLLoss()
        self.mse = nn.MSELoss()

    def predict(self,embeds1,embeds2):
        basis_outputs = []
        for i in range(0,self.num_weights):
            index = Variable(th.LongTensor([i])).to(device)
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim, self.embed_dim)
            u_Q = th.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q*embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = th.stack(basis_outputs,dim=1)
        logit = th.matmul(basis_outputs,self.weight_scalars)
        outputs = F.log_softmax(logit,dim=1)
        preds = 0
        for j in range(0,self.num_relations):
            index = Variable(th.LongTensor([j])).to(device)
            ''' j+1 because of zero index '''
            preds += (j+1)*th.exp(th.index_select(outputs, 1,index))
        return preds

    def forward(self, embeds1, embeds2, rels):
        basis_outputs = []
        for i in range(0,self.num_weights):
            index = Variable(th.LongTensor([i])).to(device)
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim, self.embed_dim)
            u_Q = th.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q*embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = th.stack(basis_outputs,dim=1)
        logit = th.matmul(basis_outputs,self.weight_scalars)
        outputs = F.log_softmax(logit,dim=1)
        log_probs = th.gather(outputs,1,rels.unsqueeze(1))
        loss = self.nll(outputs,rels)
        preds = 0
        for j in range(0,self.num_relations):
            index = Variable(th.LongTensor([j])).to(device)
            ''' j+1 because of zero index '''
            preds += (j+1)*th.exp(th.index_select(outputs, 1,index))
        return loss,preds

class SimpleGCMC(nn.Module):
    """
    This is the AutoEncoder model from the Flexible-Compositional-Fairness paper
    - this is used to provide discriminators with vectorized user embeddings. 
    - the discriminating models then predict a sensitive user attribute given the output embeddings
    
    Note:
    this model does not require any additional decoding models. There is already a prebuilt decoder model (SharedBilinearDecoder)
    """
    def __init__(self, decoder, embed_dim, num_ent, p , encoder=None, attr_filter=None):
        super(SimpleGCMC, self).__init__()
        self.attr_filter = attr_filter
        self.decoder = decoder
        self.batchnorm = nn.BatchNorm1d(embed_dim)
        self.p = p
        if encoder is None:
            r = 6 / np.sqrt(embed_dim)
            self.encoder = nn.Embedding(num_ent, embed_dim)
            self.encoder.weight.data.uniform_(-r, r).renorm_(p=2, dim=1, maxnorm=1)
        else:
            self.encoder = encoder

    def encode(self, nodes, filters=None):
        embs = self.encoder(nodes)
        embs = self.batchnorm(embs)
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                embs = apply_filters_gcmc(embs,filters)
        return embs

    def predict_rel(self,heads,tails_embed,filters=None):
        with th.no_grad():
            head_embeds = self.encode(heads)
            if filters is not None:
                constant = len(filters) - filters.count(None)
                if constant !=0:
                    head_embeds = apply_filters_gcmc(head_embeds,filters)
            preds = self.decoder.predict(head_embeds,tails_embed)
        return preds

    def forward(self, pos_edges, weights=None, return_embeds=False, filters=None):
        pos_head_embeds = self.encode(pos_edges[:,0])
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                pos_head_embeds = apply_filters_gcmc(pos_head_embeds,filters)
        pos_tail_embeds = self.encode(pos_edges[:,-1])
        rels = pos_edges[:,1]
        loss, preds = self.decoder(pos_head_embeds, pos_tail_embeds, rels)
        if return_embeds:
            return loss, preds, pos_head_embeds, pos_tail_embeds
        else:
            return loss, preds

    def save(self, fn):
        th.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(th.load(fn))

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, out_dim, activation, num_layers=2, dropout=0.3, relu = 0.2, verbose=False):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim =  out_dim
        self.relu=nn.LeakyReLU(relu)
        self.dropout=nn.Dropout(dropout)
        self.activation = activation

        layers =[] 
        for layer in range(num_layers):
            if layer == 0:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim*2, bias=True))
                layers.append(self.relu)
                layers.append(self.dropout)
            elif layer == (num_layers-1):
                layers.append(nn.Linear(self.hidden_dim*(layer*2), self.out_dim, bias=True))
            else:
                layers.append(nn.Linear(self.hidden_dim*(layer*2), self.hidden_dim*((layer+1)*2), bias=True))
                layers.append(self.relu)
                layers.append(self.dropout)
        self.model=nn.Sequential(*layers)
        
        if verbose:
            print(self.model)

    def forward(self, x):
        return self.activation(self.model(x)).squeeze()        

