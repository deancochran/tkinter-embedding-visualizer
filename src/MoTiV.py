'''
Hello this is a incomplete and under developed file. The current class is under massive change. 
It is not recommended that it be used for the prefair pipeline yet.
'''

import torch as th
import os
import zipfile
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, download_url, HeteroData

# ----------------- PYTORCH GEO HETEROGRAPHS -----------------

class MoTiV(Dataset):

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
            download_url(url, self.raw_dir)

    def process(self):
        # ----------------- INIT GRAPH -----------------
        data = HeteroData()


        # ----------------- LOADING DATAFRAMES -----------------


        # ----------------- PRE FILTERING/TRANSFORMING DATA -----------------

        # if self.pre_filter is not None:
        #     # data_list = [data for data in data_list if self.pre_filter(data)]
        #     raise Warning('pre_filter not implemented')

        # if self.pre_transform is not None:
        #     # data_list = [data for data in data_list if self.pre_filter(data)]
        #     raise Warning('pre_transform not implemented')

        # ----------------- NODE INFORMATION -----------------



        # ----------------- EDGE INFORMATION -----------------


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
        data = th.load(os.path.join(self.processed_dir, "motiv.pt"))
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
