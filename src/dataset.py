import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from torch import LongTensor

# ----------------- PYTORCH DATASET FOR TRAINING AND TESTING-----------------

class CustomDataset(Dataset):
    def __init__(self, data_split, prefetch_gpu=False):
        self.prefetch_gpu = prefetch_gpu
        self.dataset = np.ascontiguousarray(data_split)

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]
    def shuffle(self):
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()
        self.dataset = LongTensor(np.ascontiguousarray(np.random.shuffle(self.dataset)))
        if self.prefect_gpu:
            self.dataset = self.dataset.cuda().contiguous()

# ----------------- TABULAR DATASET -----------------

def load_ML100K(data_dir):
    """
    Description:
        the load_ML100k function when passed the correct directory will return all of the data from the Movie Lens 100k Dataset [http://www.movielens.org/]

    Parameters:
        data_dir=the root directory path of the dataset files

    Returns:
        a collection of pandas Dataframes (train_ratings,test_ratings,users,movies)
    """
    ratings = pd.read_csv(f'{data_dir}u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'], encoding='latin-1')
    # shuffled_ratings = ratings.sample(frac=1).reset_index(drop=True)
    # train_cutoff_row = int(np.round(len(shuffled_ratings)*0.9))
    # train_ratings = shuffled_ratings[:train_cutoff_row]
    # test_ratings = shuffled_ratings[train_cutoff_row:]
    # train_ratings=train_ratings.reindex(columns=["user_id","rating","movie_id"])-1
    # test_ratings=test_ratings.reindex(columns=["user_id","rating","movie_id"])-1
    ratings=ratings.reindex(columns=["user_id","rating","movie_id"])-1

    users = pd.read_csv(f'{data_dir}u.user', sep="|", encoding='latin-1', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    users.user_id = users.user_id.astype(np.int64)
    users['user_id'] = users['user_id'] - 1

    movies = pd.read_csv(f'{data_dir}u.item', sep="|", encoding='latin-1', header=None, parse_dates=True, 
            names=['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    movies.movie_id = movies.movie_id.astype(np.int64)
    movies['movie_id'] = movies['movie_id'] - 1

    return ratings,users,movies
