import pandas as pd
import numpy as np

# def load_ML1M(data_dir):
#     """
#     Description:
#     the load_ML1M function when passed the correct directory will return all of the data from the Movie Lens 1M Dataset [http://www.movielens.org/]

#     Parameters:
#     data_dir=the root directory path of the dataset files

#     Returns:
#     a collection of pandas Dataframes (train_ratings,test_ratings,users,movies)
#     """
#     r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
#     ratings = pd.read_csv(f'{data_dir}ratings.dat', sep='::', names=r_cols,encoding='latin-1')
#     shuffled_ratings = ratings.sample(frac=1).reset_index(drop=True)
#     train_cutoff_row = int(np.round(len(shuffled_ratings)*0.9))
#     train_ratings = shuffled_ratings[:train_cutoff_row]
#     test_ratings = shuffled_ratings[train_cutoff_row:]
#     u_cols = ['user_id','sex','age','occupation','zip_code']
#     m_cols = ['movie_id','title','genre']
#     users = pd.read_csv(f'{data_dir}users.dat', sep='::', names=u_cols,encoding='latin-1', parse_dates=True)
#     movies = pd.read_csv(f'{data_dir}movies.dat', sep='::', names=m_cols,encoding='latin-1', parse_dates=True)

#     train_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
#     train_ratings_matrix = train_ratings.pivot_table(index=['movie_id'],columns=['user_id'],values='rating').reset_index(drop=True)
#     test_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
#     columnsTitles=["user_id","rating","movie_id"]
#     train_ratings=train_ratings.reindex(columns=columnsTitles)-1
#     test_ratings=test_ratings.reindex(columns=columnsTitles)-1
#     users.user_id = users.user_id.astype(np.int64)
#     movies.movie_id = movies.movie_id.astype(np.int64)
#     users['user_id'] = users['user_id'] - 1
#     movies['movie_id'] = movies['movie_id'] - 1

#     return train_ratings,test_ratings,users,movies


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
    shuffled_ratings = ratings.sample(frac=1).reset_index(drop=True)
    train_cutoff_row = int(np.round(len(shuffled_ratings)*0.9))
    train_ratings = shuffled_ratings[:train_cutoff_row]
    test_ratings = shuffled_ratings[train_cutoff_row:]
    train_ratings=train_ratings.reindex(columns=["user_id","rating","movie_id"])-1
    test_ratings=test_ratings.reindex(columns=["user_id","rating","movie_id"])-1

    users = pd.read_csv(f'{data_dir}u.user', sep="|", encoding='latin-1', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    users.user_id = users.user_id.astype(np.int64)
    users['user_id'] = users['user_id'] - 1

    movies = pd.read_csv(f'{data_dir}u.item', sep="|", encoding='latin-1', header=None, parse_dates=True, 
            names=['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    movies.movie_id = movies.movie_id.astype(np.int64)
    movies['movie_id'] = movies['movie_id'] - 1

    return train_ratings,test_ratings,users,movies


