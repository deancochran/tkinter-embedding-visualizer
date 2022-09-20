import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from scipy import stats
import itertools as it
import torch
from sklearn.manifold import TSNE
from torch_geometric.nn import MetaPath2Vec

def get_metapath2vec_embbeddings(edge_index, metapath, device, embedding_dim=64, walk_length=25, context_size=7,walks_per_node=5, num_negative_samples=5,sparse=True, batch_size=128, shuffle=True, num_workers=6,lr=0.01, epochs = 25):
    metapath2vec = MetaPath2Vec(edge_index, embedding_dim, metapath, walk_length, context_size, walks_per_node, num_negative_samples, sparse=True).to(device)
    loader = metapath2vec.loader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(metapath2vec.parameters()), lr)
    metapath2vec.train()
    for epoch in range(0, epochs):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = metapath2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            print('\r',f'Training metapath2vec embedding representations Epoch: {epoch+1} of {epochs}, Step: {i + 1:.3f}/{len(loader)}, 'f'Loss: {loss:.3f}', end=' ')
    return {node_type: metapath2vec(node_type).detach().cpu() for node_type in metapath2vec.num_nodes_dict}

def get_tsne_emb(embeddings, n_components=2, init='random', perplexity=5, return_tensor=False):
    if return_tensor:
        return torch.tensor(TSNE(n_components=n_components, init=init, perplexity=perplexity).fit_transform(embeddings))
    else:
        return TSNE(n_components=n_components, init=init, perplexity=perplexity).fit_transform(embeddings)

def rf_discriminator_test(X,y):

    # splitting X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2)

    # # training a ML classifier
    feature_names = [f"X_{[i]}" for i in range(X.shape[1])]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)
    
    # clf = DecisionTreeClassifier()
    y_pred = forest.predict(X_test)

    # creating a figure for plotting
    fig = plt.figure(figsize=(10,5))
    fig.suptitle('Confusion Matrix and Entity Embedding Plot')
    ax1= fig.add_subplot(121)

    # creating a confusion matrix
    cm =  confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True).plot()
    ax1.set_title("Confusion Matrix")
    ax1.set_ylabel("y_true")
    ax1.set_xlabel("y_pred")

    # creating feature importances
    # ax2 = fig.add_subplot(122)
    # importances = forest.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    # forest_importances = pd.Series(importances, index=feature_names)
    # forest_importances.plot.bar(yerr=std, ax=ax2)
    # ax2.set_title("Feature importances")
    # ax2.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    
    # Embedding plot
    # fig = plt.figure(figsize=(6,4))
    if X.shape[1] > 2:
        ax_emb = fig.add_subplot(122, projection='3d')
        ax_emb.scatter(X[:,0],X[:,1],X[:,2], c=y, label=y.numpy())
        ax_emb.set_title("Embedding Plot")
        ax_emb.set_zlabel("X[:,2]")
        ax_emb.set_ylabel("X[:,1]")
        ax_emb.set_xlabel("X[:,0]")
    else:
        ax_emb = fig.add_subplot(122)# create axes
        ax_emb.scatter(X[:,0],X[:,1], c=y.numpy())
        ax_emb.set_title("Embedding Plot")
        ax_emb.set_ylabel("X[:,1]")
        ax_emb.set_xlabel("X[:,0]")
    plt.show()
    
    print(classification_report(y_test, y_pred))

def perform_2sided_ks_tests(data: pd.DataFrame, distribution_attr: str, sensitive_attr:str):
    """
    Description:
    perform_ks_tests returns a set of all permutations of the pairs of sensitive attributes,
    and their 2 sample KS statistics of the specified distribution in question

    For a 2-sided KS test, the null hypothesis is that the two distributions are identical, 
    F(x)=G(x) for all x; the alternative is that they are not identical.

    If the p-val < .05 reject the null hypothesis

    Arguments:
    data - pandas Dataframe obj containing both the distribution_attr and the sensitive_attr in the column names
    distribution_attr -  this is a float valued column in the data argument
    sensitive_attr - this is a categorical string valued column in the data argument

    Returns:
    dict() - {(attr_a, attr_b): (k_stat, p_val),...} for all combinations of categories in sensitive attribute column

    """
    results = {}
    sensitive_attr_categories = data[sensitive_attr].unique()
    assert len(sensitive_attr_categories) > 1
    for combination in it.combinations(sensitive_attr_categories, 2):
        (attr_a, attr_b) = combination
        attr_a_data = data[data[sensitive_attr]==attr_a][distribution_attr]
        attr_b_data = data[data[sensitive_attr]==attr_b][distribution_attr]
        results[combination] = stats.kstest(attr_a_data, attr_b_data)

    return results

def get_wasserstein_dist(data, distribution_attr, sensitive_attr, attr_a, attr_b):
    """Description:
    get_wasserstein_dist computes the distributional distance between two samples. 

    Arguments:
    data - pandas Dataframe obj containing both the distribution_attr and the sensitive_attr in the column names
    distribution_attr -  this is a float valued column in the data argument
    sensitive_attr - this is a categorical string valued column in the data argument
    attr_a = string name of the first sample (a sensitive attribute label)
    attr_b = string name of the second sample (a sensitive attribute label)

    Returns:
    distance: Int
    """
    attr_a_data = data[data[sensitive_attr]==attr_a][distribution_attr]
    attr_b_data = data[data[sensitive_attr]==attr_b][distribution_attr]
    distance = stats.wasserstein_distance(attr_a_data, attr_b_data)

    return distance


def analyze_2sided_ks_results(data: pd.DataFrame, distribution_attr: str, sensitive_attr:str):
    """
    Description:
    analyze_2sided_ks_results uses the output from perform_2sided_ks_tests to identify pairs of sensitive attributes,
    of the indicated distribution_attr, that are statistically different. 
    (looks for pairs whose p-val < .05 == reject the null hypothesis)

    Arguments:
    data - pandas Dataframe obj containing both the distribution_attr and the sensitive_attr in the column names
    distribution_attr -  this is a float valued column in the data argument
    sensitive_attr - this is a categorical string valued column in the data argument

    Returns:
    dict() - {(attr_a, attr_b): (k_stat, p_val),...} for all combinations of categories in sensitive attribute column

    """
    output={}
    results = perform_2sided_ks_tests(data, distribution_attr, sensitive_attr)
    for (attr_a, attr_b), (k_stat, p_val) in results.items():
        if p_val<.05:
            output[(attr_a, attr_b)]=get_wasserstein_dist(data, distribution_attr, sensitive_attr, attr_a, attr_b)
    return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}

def analyze_sensitive_attr_distributions(data: pd.DataFrame, distribution_attr_list: list, sensitive_attr_list: list):
    """
    Description:
    analyze_sensitive_attr_distributions is a function that utilizes analyze_2sided_ks_results and perform_2sided_ks_tests
    to extend the opportunities of findings by aggregation all statistically different pairings of sensitive attributes, 
    for all the possible distributions that are given 
    
    Arguments:
    data - pandas Dataframe obj containing both the distribution_attr and the sensitive_attr in the column names
    distribution_attr_list - list of column names
    sensitive_attr_list - list of column names

    Returns:
    
    """
    results={}
    for (distr_attr, sens_attr) in it.product(distribution_attr_list,sensitive_attr_list): 
        output=analyze_2sided_ks_results(data,distr_attr,sens_attr)
        if len(output) > 0:
            results[(distr_attr,sens_attr)]=output
    return results

def get_optimizer(name:str, parameters):
    if name == "adam":
        return torch.optim.Adam(params=parameters)
    else:
        raise Exception(f'Optimizer {name} has not been added yet')
def get_activation(name:str):
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'softmax':
        return nn.Softmax(dim=1)
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
        return roc_auc_score(y, y_hat, average='micro')
    else:
        return roc_auc_score(y, y_hat)

def plot_train_test_embedding(epochs,train_loss_vals,test_loss_vals, train_auc_vals, test_auc_vals, embeddings, embedding_labels):
    fig = plt.figure(figsize=(10,10))
    train_ax = plt.subplot(221)
    train_ax.plot(list(range(epochs)), train_loss_vals, label='loss')
    train_ax.plot(list(range(epochs)), train_auc_vals, label='auc')
    train_ax.set_ylim(0,1)
    train_ax.legend()

    test_ax = plt.subplot(222)
    test_ax.plot(list(range(epochs)), test_loss_vals, label='loss')
    test_ax.plot(list(range(epochs)), test_auc_vals, label='auc')
    test_ax.set_ylim(0,1)
    test_ax.legend()

    if embeddings.shape[1] > 2:
        projection = '3d'
    else:
        projection = '2d'
    # Embedding plot
    if projection == '3d':
        ax_emb = plt.subplot(212, projection=projection)
        ax_emb.scatter(embeddings[:,0],embeddings[:,1],embeddings[:,2], c=embedding_labels)
    else:
        ax_emb = plt.subplot(212)# create axes
        ax_emb.scatter(embeddings[:,0],embeddings[:,1], c=embedding_labels)
    plt.show()
    
    
    
def plot_train_test_embedding(epochs,train_loss_vals,test_loss_vals, train_auc_vals, test_auc_vals, embeddings, embedding_labels):
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('Training/Testing Visual and Entity Embedding Plot')
    train_ax = plt.subplot(221)
    train_ax.plot(list(range(epochs)), train_loss_vals, label='loss')
    train_ax.plot(list(range(epochs)), train_auc_vals, label='auc')
    train_ax.set_ylim(0,1)
    # train_ax.set_ylabel('%')
    train_ax.set_xlabel('Epochs')
    train_ax.legend()

    test_ax = plt.subplot(222)
    test_ax.plot(list(range(epochs)), test_loss_vals, label='loss')
    test_ax.plot(list(range(epochs)), test_auc_vals, label='auc')
    test_ax.set_ylim(0,1)
    test_ax.set_ylabel('%')
    test_ax.set_xlabel('Epochs')
    test_ax.legend()

    if embeddings.shape[1] > 2:
        projection = '3d'
    else:
        projection = '2d'
    # Embedding plot
    if projection == '3d':
        ax_emb = plt.subplot(212, projection=projection)
        ax_emb.scatter(embeddings[:,0],embeddings[:,1],embeddings[:,2], c=embedding_labels)
        ax_emb.set_title("Embedding Plot")
        ax_emb.set_zlabel("X[:,2]")
        ax_emb.set_ylabel("X[:,1]")
        ax_emb.set_xlabel("X[:,0]")
    else:
        ax_emb = plt.subplot(212)# create axes
        ax_emb.scatter(embeddings[:,0],embeddings[:,1], c=embedding_labels)
        ax_emb.set_title("Embedding Plot")
        ax_emb.set_ylabel("X[:,1]")
        ax_emb.set_xlabel("X[:,0]")
    
    plt.show()