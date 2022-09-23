"""
Tkinter Demo:
This is a demostration of the capabilities that Tkinter can offer users when the specified functions are correctly implemented. This is NOT production ready. 
This demonstration will load a program that can be interacted with...

Goal:
To provide a user a visual and interactive way to filter datasets, select there methods for generating embeddings for node types, and then
receive information on the vulnerabilities of the data used for training/testing a prediction classifier on a sensitive attribute.

Design Improvements to be made:
Without mentioning the bare aspects of the program here is a list of features that need to be implemented..

- incorporating a refresh function that will restart the program without changing the layout of the program
- a filtering method to allow the user to make attribute splits on the selected data before training
- incorperating text labels of the output data used to make the plots
- incorperating a prgress bar for future encoding/training mehtods that could be selected 
- preventing automatic resizing of window, it would be simpler to have a fixed ratio of the program
- possibly removing the matplotlib graph tool bar
-...
"""

import tkinter as tk
from tkinter import StringVar, OptionMenu, Button
import os, sys
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import torch as th
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import tqdm
from dataset import load_ML100K
from dataset import CustomDataset
from ml100k import ML100k
from model import Discriminator
device = 'cuda' if th.cuda.is_available() else 'cpu'



# ----------------- UTILITY FUNCTIONS FOR AUTOMATED DATASET DISCRIMINATION TESTING-----------------
    
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

    
# ----------------- TKINTER PREFAIR APP -----------------
class PrEFairApp(tk.Tk):
    """
    Description:
    PreFairApp is a interactive visualization for the PrEFair Project.
    Parameters:
    """
    def __init__(self):
        super().__init__()
        self.data_root = '../data'
        self.init_menu()

    def init_menu(self):
        """
        Assumptions that need to be tested:
        1. The data_root path exists
        2. Their is at minimum, 1 dataset folder in the data_root directory
        3. Folder contains and dataset that cane be processed by the utility functions in this repository
        """

        datasets = os.listdir(self.data_root)
        self.dataset_var=StringVar(value='Select a Dataset')
        
        self.dataset_menu = OptionMenu(self, self.dataset_var,  *['datasets'], command=self.update_ents)
        self.dataset_menu.pack()

        self.entity_var=StringVar(value='Select an Entity')
        self.entity_menu = OptionMenu(self, self.entity_var,  *['None'], command=self.update_feats)
        self.entity_menu.configure(state='disabled')
        self.entity_menu.pack()

        self.feature_var=StringVar(value='Select a Feature')
        self.feature_menu = OptionMenu(self, self.feature_var,  *['None'], command=self.update_encoder)
        self.feature_menu.configure(state='disabled')
        self.feature_menu.pack()

        self.encoding_var=StringVar(value='Select an Encoding Method')
        self.encoding_menu = OptionMenu(self, self.encoding_var,  *['None'], command=self.update_decoder)
        self.encoding_menu.configure(state='disabled')
        self.encoding_menu.pack()

        self.decoding_var=StringVar(value='Select an Decoding Method')
        self.decoding_menu = OptionMenu(self, self.decoding_var,  *['None'], command=self.enable_train)
        self.decoding_menu.configure(state='disabled')
        self.decoding_menu.pack()

        self.train_btn = Button(self, text='Train', command=self.train)
        self.train_btn.configure(state='disabled')
        self.train_btn.pack()

        self.quit_btn = Button(self, text='Quit',command=self.destroy)
        self.quit_btn.pack()

    def update_ents(self, dataset_name):
        if dataset_name != 'Select a Dataset':
            print(f"{dataset_name}")
            ent_vals = entity_vals(self.data_root, dataset_name)
            print(ent_vals)
            self.entity_menu.configure(state='active')

    def update_feats(self, entity_name):
        print(f"{entity_name}")
        dataset_name = self.dataset_var.get()
        feat_vals = feature_vals(self.data_root, dataset_name, entity_name)
        print(feat_vals)
        self.feature_menu.configure(state='active')

    def update_encoder(self, feature_name):
        print(f"{feature_name}")
        dataset_name = self.dataset_var.get()
        enc_vals = encoding_vals(self.data_root, dataset_name)
        print(enc_vals)
        self.encoding_menu.configure(state="active")

    def update_decoder(self, encoder_name):
        print(encoder_name)
        dataset_name = self.dataset_var.get()
        dec_vals = decoding_vals(self.data_root, dataset_name)
        print(dec_vals)
        self.decoding_menu.configure(state='active')

    def enable_train(self,decoder_name):
        print(decoder_name)
        self.train_btn.configure(state="active")

    def train(self):
        dataset_name = self.dataset_var.get()
        entity_name = self.entity_var.get()
        feature_name = self.feature_var.get()
        encoder_name = self.encoding_var.get()
        decoder_name = self.decoding_var.get()
        train_metrics, test_metrics, embeddings, labels, projection = get_discrimination_results(self.data_root, dataset_name, entity_name, feature_name, encoder_name, decoder_name)
        self.init_fig(train_metrics, test_metrics, embeddings, labels, projection)
        self.disable()

    def disable(self):
        self.dataset_menu.configure(state='disabled')
        self.entity_menu.configure(state='disabled')
        self.feature_menu.configure(state='disabled')
        self.encoding_menu.configure(state='disabled')
        self.decoding_menu.configure(state='disabled')
        self.train_btn.configure(state='disabled')

    def init_fig(self, train_metrics, test_metrics, embeddings, labels, projection=None):
        self.figure = Figure(figsize=(6, 4))  # create a figure
        self.figure_canvas = FigureCanvasTkAgg(self.figure, self)  # create FigureCanvasTkAgg object
        self.toolbar=NavigationToolbar2Tk(self.figure_canvas, self)  # create the toolbar

        # Embedding plot
        if projection is not None:
            ax_emb = self.figure.add_subplot(212, projection=projection)
            ax_emb.scatter(embeddings[:,0],embeddings[:,1],embeddings[:,2], c=labels)
        else:
            ax_emb = self.figure.add_subplot(212)# create axes
            ax_emb.scatter(embeddings[:,0],embeddings[:,1], c=labels)

        num_epochs = len(train_metrics['loss_vals'])
        # Training plot
        ax_train = self.figure.add_subplot(221)
        ax_train.plot(list(range(num_epochs)), train_metrics['loss_vals'], label='loss')
        ax_train.plot(list(range(num_epochs)), train_metrics['auc_vals'], label='auc')
        ax_train.set_ylim(0,1)
        ax_train.legend()

        # Testing plot
        ax_test = self.figure.add_subplot(222)
        ax_test.plot(list(range(num_epochs)), test_metrics['loss_vals'], label='loss')
        ax_test.plot(list(range(num_epochs)), test_metrics['auc_vals'], label='auc')
        ax_test.set_ylim(0,1)
        ax_test.legend()
        self.figure.suptitle('User Attribute Prediciton with Emeddings:')
        self.emb_visual = self.figure_canvas.get_tk_widget()
        self.emb_visual.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

if __name__ == '__main__':
    app = PrEFairApp()
    app.mainloop()
