import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import torch as th
from torch.autograd import Variable
from torch.utils.data import DataLoader
import tqdm

from dataloader import load_ML100K
from dataset import CustomDataset
from model import Discriminator, Encoder
from plot import PrEFairApp
from utils import activation, criterion, optimizer, roc_auc
device = 'cuda' if th.cuda.is_available() else 'cpu'

train_ratings,test_ratings,users_df,movies_df = load_ML100K('../data/ml-100k/raw/ml-100k/')
y_true=np.ascontiguousarray(users_df['gender'].apply(lambda x: 0 if x == 'M' else 1))
users=np.asarray(list(set(users_df['user_id'])))
num_entities = len(users)
cutoff = int(np.round(len(users)*.8))
train_users = users[:cutoff]
test_users = users[cutoff:]
user_train_set = CustomDataset(train_users)
user_test_set = CustomDataset(test_users)

HID_DIM=32
ACTIVATION = 'sigmoid'
OPTIMIZER = 'adam'
CRITERION = 'BCELoss'
OUT_DIM=1
EPOCHS = 100
BATCH_SIZE=512
NUM_WORKERS=1

discriminator=Discriminator(hidden_dim = HID_DIM, out_dim=OUT_DIM, activation=activation(ACTIVATION))
train_loader = DataLoader(user_train_set, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
test_loader = DataLoader(user_test_set, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
loss_fn = criterion(CRITERION)
opt = optimizer(OPTIMIZER,discriminator.parameters())
usermodel=Encoder(hidden_dim=32, num_ent=num_entities)

def test_discriminator(test_loader, encoder,discriminator, y_true):
    probs_list, labels_list = [], []
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).to(device)
        p_batch_emb = usermodel(p_batch_var.detach())
        y_hat = discriminator(p_batch_emb)
        y = Variable(th.FloatTensor(y_true[p_batch_var.cpu()])).to(device)
        probs_list.append(y_hat)
        labels_list.append(y)
    Y = th.cat(labels_list,0)
    Y_hat = th.cat(probs_list,0)
    loss = loss_fn(Y_hat,Y)
    auc = roc_auc(Y.data.cpu().numpy(), Y_hat.data.cpu().numpy())
    return loss, auc

usermodel.eval()
dataloader=tqdm.tqdm(range(1, EPOCHS+1))
train_loss_vals, test_loss_vals = [], []
train_auc_vals, test_auc_vals = [], []
for epoch in dataloader:
    probs_list, labels_list = [], []
    for p_batch in train_loader:
        p_batch_var = Variable(p_batch).to(device)
        p_batch_emb = usermodel(p_batch_var.detach())
        y_hat = discriminator(p_batch_emb)
        y = Variable(th.FloatTensor(y_true[p_batch_var.cpu()])).to(device)
        probs_list.append(y_hat)
        labels_list.append(y)
        loss = loss_fn(y_hat,y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    Y = th.cat(labels_list,0)
    Y_hat = th.cat(probs_list,0)
    train_auc = roc_auc(Y.data.cpu().numpy(), Y_hat.data.cpu().numpy())
    train_loss = loss_fn(Y_hat, Y)
    test_loss, test_auc = test_discriminator(test_loader, usermodel, discriminator, y_true)
    train_loss_vals.append(train_loss.item())
    test_loss_vals.append(test_loss.item())
    train_auc_vals.append(train_auc.item())
    test_auc_vals.append(test_auc.item())
    # dataloader.set_description(f'Train Loss: {train_loss}, Train AUC: {train_auc}, Test Loss: {test_loss}, Test AUC: {test_auc}')
app=PrEFairApp()
user_embs = usermodel(th.tensor(users)).detach()
data = TSNE(n_components=3, init='random', perplexity=3).fit_transform(user_embs)
train_metrics = {
        'loss_vals':train_loss_vals,
        'auc_vals':train_auc_vals
        }
test_metrics = {
        'loss_vals':test_loss_vals,
        'auc_vals':test_auc_vals
        }
app.init_fig(train_metrics, test_metrics, data, y_true, projection = '3d')
app.mainloop()
