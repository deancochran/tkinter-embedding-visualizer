import torch as th
import os
import zipfile
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, download_url, HeteroData

# ----------------- PYTORCH GEO HETEROGRAPHS -----------------

class MoTiV(Dataset):

    """
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        # Download to `self.raw_dir`.
        urls={
            'activities':'https://zenodo.org/record/4027465/files/activities.csv',
            'experience_factors':'https://zenodo.org/record/4027465/files/experience_factors.csv',
            'legs':'https://zenodo.org/record/4027465/files/legs.csv',
            'legs_coordinates':'https://zenodo.org/record/4027465/files/legs_coordinates.csv',
            'mots':'https://zenodo.org/record/4027465/files/mots.csv',
            'purposes':'https://zenodo.org/record/4027465/files/purposes.csv',
            'trips':'https://zenodo.org/record/4027465/files/trips.csv',
            'user_details':'https://zenodo.org/record/4027465/files/user_details.csv',
            'user_generic_worthwhileness_values':'https://zenodo.org/record/4027465/files/user_generic_worthwhileness_values.csv',
            'user_specific_worthwhileness_values':'https://zenodo.org/record/4027465/files/user_specific_worthwhileness_values.csv',
            'weather_legs':'https://zenodo.org/record/4027465/files/weather_legs.csv',
            'weather_raw':'https://zenodo.org/record/4027465/files/weather_raw.csv',
            'worthwhileness_elements_from_trips':'https://zenodo.org/record/4027465/files/worthwhileness_elements_from_trips.csv',
        }
        for name, url in urls.items():
            zip_name = url.split('/')[-1].split('.')[0]
            path = download_url(url, self.raw_dir)
            # with zipfile.ZipFile(path, 'r') as zip_ref:
            #     zip_ref.extractall(self.raw_dir)
        #     self.raw_ml100k_dir=self.raw_dir+'/'+zip_name

    def process(self):
        # ----------------- INIT GRAPH -----------------
        data = HeteroData()

        # ----------------- LOADING DATAFRAMES -----------------

        # movies = pd.read_csv(f'{self.raw_ml100k_dir}/u.item', sep="|", encoding='latin-1', header=None, parse_dates=True,
        #             names=['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

        # self.movie_mapping = {idx: int(i) for i, idx in enumerate(movies['movie_id'])}

        # users = pd.read_csv(f'{self.raw_ml100k_dir}/u.user', sep="|", encoding='latin-1', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
        # self.user_mapping = {idx: int(i) for i, idx in enumerate(users['user_id'])}

        # ratings = pd.read_csv(f'{self.raw_ml100k_dir}/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')
        # src = [self.user_mapping[idx] for idx in ratings['user_id']]
        # dst = [self.movie_mapping[idx] for idx in ratings['movie_id']]


        # ----------------- PRE FILTERING/TRANSFORMING DATA -----------------

        # if self.pre_filter is not None:
        #     # data_list = [data for data in data_list if self.pre_filter(data)]
        #     raise Warning('pre_filter not implemented')

        # if self.pre_transform is not None:
        #     # data_list = [data for data in data_list if self.pre_filter(data)]
        #     raise Warning('pre_transform not implemented')

        # ----------------- NODE INFORMATION -----------------

        # # init users and add node index
        # data['user'].num_nodes = len(self.user_mapping)
        # data['user'].user_index = th.LongTensor([self.user_mapping[idx] for idx in users['user_id']])

        # # add additional user information
        # self.gender_mapping = {val: idx for idx, val in enumerate(users['gender'].unique())}
        # data['user'].gender = th.LongTensor([self.gender_mapping[val] for val in users['gender'].values])
        
        # data['user'].age = th.LongTensor([val for val in users['age'].values])
        
        # occupation_df = pd.read_csv(f'{self.raw_ml100k_dir}/u.occupation', sep="|", encoding='latin-1', header=None, names=['occupation']).reset_index().rename(columns={'index': 'occupation_id'})
        # self.occupation_mapping = {occupation_df['occupation'][i]: occupation_df['occupation_id'][i] for i in occupation_df.index}
        # data['user'].occupation = th.LongTensor([self.occupation_mapping[val] for val in users['occupation'].values])
        
        # self.zip_code_mapping={val: idx for idx, val in enumerate(users['zip_code'].unique())}
        # data['user'].zip_code = th.LongTensor([self.zip_code_mapping[val] for val in users['zip_code'].values])

        # # init movies and add node index
        # data['movie'].num_nodes = len(self.movie_mapping)
        # data['movie'].user_index = th.LongTensor([self.movie_mapping[idx] for idx in movies['movie_id']])
        # # Add Movie Title Word2Vec Embeddings Here
        # self.genre_mapping={name: idx for idx, name in enumerate(['unknown', 'Action', 'Adventure', 'Animation', "Children\'s", 'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'])}
        # data['movie'].genre=[[self.genre_mapping[k] for k in self.genre_mapping.keys() if row[k]==1] for _ , row in movies.iterrows()]

        # ----------------- EDGE INFORMATION -----------------
        # #init the user-to-movie edge and add edge index
        # data['user', 'rates', 'movie'].edge_index = th.tensor([src, dst])
        # data['user', 'rates', 'movie'].edge_rating = th.from_numpy(ratings['rating'].values).to(th.long)
        # data['user', 'rates', 'movie'].edge_timestamp = th.from_numpy(ratings['timestamp'].values).to(th.long)

        # if self.transform is not None:
        #     if type(self.transform) is not list:
        #         raise Exception('transform must be a list')
        #     for transform_name in self.transform:
        #         data = self.transform_data(data, transform_name)

        # th.save(data, os.path.join(self.processed_dir, f'ml_100k.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # use index if more than one processed file is stored in the processed directory
        # data = th.load(os.path.join(self.processed_dir, "ml_100k.pt"))
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

    # @property
    # def metapath(self):
    #     return [('user', 'rates', 'movie'), ('movie', 'rev_rates', 'user')]

if __name__ == '__main__':
    root_path='../data/MoTiV'
    # dataset = ML100k(root=root_path, transform=['undirected'], pre_transform=None, pre_filter=None)
    dataset = MoTiV(root=root_path)
    # print(dataset)
    # print(dataset.get(0))
