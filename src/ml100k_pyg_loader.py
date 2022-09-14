from typing import List
import torch as th
import os
import zipfile
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, download_url, HeteroData

class ML100k(Dataset):
    """
    ML100K is a pytorch geometric data loading class that provides more contextual information than what is provided on the PyTorch Geometric website.
    The full extent of the Movie Lens 100k dataset is utilized to store as much contextual information for the analysis of feature importance when conducting different ML tasks

    Args: 
    root (str) - the file path to a data directory 
    transform (list) - the transform list may contain the following elements: 
                    ('undirected', 'self_loops','normalize' )
                    to provide additoinal functionality to make undirected graphs, graphs with self loops, and a normalized graph respectively
    pre_transform (list) - the pre_transform list is not implemented yet
    pre_filter (list) - the transform list is not implemented yet

    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['u4.test', 'u3.test', 'u3.base', 'ub.base', 'README', 'u2.test', 'ub.test', 'u.occupation', 'u.user', 'u.item', 'ua.test', 'u4.base',
         'u.data', 'u1.test', 'u1.base', 'allbut.pl', 'u.info', 'mku.sh', 'u.genre', 'u5.base', 'u5.test', 'u2.base', 'ua.base']

    @property
    def processed_file_names(self):
        return ['ml_100k.pt', 'pre_filter.pt','pre_transform.pt']

    def download(self):
        # Download to `self.raw_dir`.
        url='https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        zip_name = url.split('/')[-1].split('.')[0]
        path = download_url(url, self.raw_dir)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
            self.raw_ml100k_dir=self.raw_dir+'/'+zip_name

    def process(self):
        # ----------------- INIT GRAPH -----------------
        data=HeteroData()


        # ----------------- LOADING DATAFRAMES -----------------

        movies = pd.read_csv(f'{self.raw_ml100k_dir}/u.item', sep="|", encoding='latin-1', header=None, parse_dates=True, 
                names=['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        movie_mapping = {idx: int(i) for i, idx in enumerate(movies['movie_id'])}
        
        users = pd.read_csv(f'{self.raw_ml100k_dir}/u.user', sep="|", encoding='latin-1', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
        user_mapping = {idx: int(i) for i, idx in enumerate(users['user_id'])}
        
        ratings = pd.read_csv(f'{self.raw_ml100k_dir}/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')
        src = [user_mapping[idx] for idx in ratings['user_id']]
        dst = [movie_mapping[idx] for idx in ratings['movie_id']]

        occupation_df = pd.read_csv(f'{self.raw_ml100k_dir}/u.occupation', sep="|", encoding='latin-1', header=None, names=['occupation']).reset_index().rename(columns={'index': 'occupation_id'})
        occupation_mapping = {occupation_df['occupation'][i]: occupation_df['occupation_id'][i] for i in occupation_df.index}


        # ----------------- PRE FILTERING/TRANSFORMING DATA -----------------
 

        if self.pre_filter is not None:
            # data_list = [data for data in data_list if self.pre_filter(data)]
            raise Warning('pre_filter not implemented')

        if self.pre_transform is not None:
            # data_list = [data for data in data_list if self.pre_filter(data)]
            raise Warning('pre_transform not implemented')

        
        # ----------------- NODE INFORMATION -----------------
        
        # init users and add node index
        data['user'].num_nodes = len(user_mapping)
        data['user'].user_index = th.LongTensor([user_mapping[idx] for idx in users['user_id']])

        # add additional user information
        gender_mapping = {val: idx for idx, val in enumerate(users['gender'].unique())}
        data['user'].gender = th.LongTensor([gender_mapping[val] for val in users['gender'].values])
        data['user'].age = th.LongTensor([val for val in users['age'].values])
        occupation_mapping={occupation_df['occupation'][i]: occupation_df['occupation_id'][i] for i in range(occupation_df.shape[0])}        
        data['user'].occupation = th.LongTensor([occupation_mapping[val] for val in users['occupation'].values])
        zip_code_mapping={val: idx for idx, val in enumerate(users['zip_code'].unique())}
        data['user'].zip_code = th.LongTensor([zip_code_mapping[val] for val in users['zip_code'].values])
        

        # init movies and add node index
        data['movie'].num_nodes = len(movie_mapping)
        data['movie'].user_index = th.LongTensor([movie_mapping[idx] for idx in movies['movie_id']])
        ############ Add Movie Title Word2Vec Embeddings Here
        genre_mapping={name: idx for idx, name in enumerate(['unknown', 'Action', 'Adventure', 'Animation', "Children\'s", 'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'])}
        data['movie'].genre=[[genre_mapping[k] for k in genre_mapping.keys() if row[k]==1] for _ , row in movies.iterrows()]

        # ----------------- EDGE INFORMATION -----------------

        #init the user-to-movie edge and add edge index
        data['user', 'rates', 'movie'].edge_index = th.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_rating = th.from_numpy(ratings['rating'].values).to(th.long)
        data['user', 'rates', 'movie'].edge_timestamp = th.from_numpy(ratings['timestamp'].values).to(th.long)

        if self.transform is not None:
            if type(self.transform) is not list:
                raise Exception('transform must be a list')
            for transform_name in self.transform:
                data = self.transform_data(data, transform_name)

        th.save(data, os.path.join(self.processed_dir, f'ml_100k.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # use index if more than one processed file is stored in the processed directory
        data = th.load(os.path.join(self.processed_dir, f'ml_100k.pt'))
        return data

    def transform_data(self, data, transform_name):
        if transform_name == 'undirected':
            data = T.ToUndirected()(data)
        elif transform_name == 'self_loops':
            data = T.AddSelfLoops()(data)
        elif transform_name == 'normalize':
            data = T.NormalizeFeatures()(data)
        else:
            raise Exception('Not implemented yet')
        return data
    @property
    def metapath(self):
        return [
        ('user', 'rates', 'movie'),
        ('movie', 'rev_rates', 'user')
        ]


if __name__ == '__main__':
    root_path='../../data/ml-100k'
    dataset = ML100k(root=root_path, transform=['undirected'], pre_transform=None, pre_filter=None)
    print(dataset)
    print(dataset.get(0))