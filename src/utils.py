import torch as th
import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
def get_optimizer(name:str, parameters):
    if name == "adam":
        return th.optim.Adam(params=parameters)
    else:
        raise Exception(f'Optimizer {name} has not been added yet')
def get_activation(name:str):
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'softmax':
        return nn.Softmax()
    else:
        raise Exception(f'Activation function {name} has not been added yet')

def get_criterion(name:str):
    if name == 'BCELoss':
        return nn.BCELoss()
    if name == 'CELoss':
        return nn.CrossEntropyLoss()
    else:
        raise Exception(f'Criterion function {name} has not been added yet')

def get_roc_auc(y, y_hat, name:str=None):
    if name=='multi':
        y = np.asarray(y).squeeze()
        y_hat = np.asarray(y_hat).squeeze()
        lb = LabelBinarizer()
        lb.fit(y)
        y = lb.transform(y)
        return roc_auc_score(y, y_hat, average='macro',multi_class='ovo')
    else:
        return roc_auc_score(y, y_hat)

def entity_vals(data_dir, dataset_name):
    print("getting ents for",dataset_name)
    return ['user','movie']

def feature_vals(data_dir, dataset_name, entity_name):
    print('getting feats for combination:',dataset_name, entity_name)
    return ['age','gender','occupation']

def encoding_vals(data_dir, dataset_name):
    print('encoding vals for ...',dataset_name)
    return ['metapath2vec','MLP']

def decoding_vals(data_dir, dataset_name):
    print('decoding vals for ...',dataset_name)
    return ['MLP','DecisionTreeClassifier']

def get_discrimination_results(data_dir, dataset_name, entity_name, feature_name, encoder_name, decoder_name):
    print(f'training with dataset:{dataset_name}, entities:{entity_name}, features:{feature_name}, encoder:{encoder_name}, decoder:{decoder_name}')
    y = np.random.uniform(0,1,1000)
    x = np.random.uniform(0,1,1000)
    labels=np.random.randint(0, 2, 1000)
    emb=np.vstack([x,y]).T
    train_loss = np.arange(0,1,.01)
    test_loss = np.arange(0,1,.01)
    if len(emb[0])>2:
        return {'loss_vals':train_loss, 'auc_vals':test_loss},{'loss_vals':train_loss, 'auc_vals':test_loss}, emb, labels, '3d'
    else:
        return {'loss_vals':train_loss, 'auc_vals':test_loss},{'loss_vals':train_loss, 'auc_vals':test_loss}, emb, labels, None
