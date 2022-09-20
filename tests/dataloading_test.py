import sys
import os
 
# getting the name of the directory
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name
parent = os.path.dirname(current)
# adding the parent directory
sys.path.append(parent)
# setting the parent directory
os.chdir(parent)

# importing from parent
from src import dataset

def test_ML100K(ml_100k_path):
    '''
    ML-100k tests:
    - test loading function
    - test shape of training/testing rating data frames
    - test shape of user, and item data frames
    ''' 
    print('Testing ml-100k')
    assert os.listdir(ml_100k_path) == ['u4.test', 'u3.test', 'u3.base', 'ub.base', 'README', 'u2.test', 'ub.test', 'u.occupation', 'u.user', 'u.item', 'ua.test',\
     'u4.base', 'u.data', 'u1.test', 'u1.base', 'allbut.pl', 'u.info', 'mku.sh', 'u.genre', 'u5.base', 'u5.test', 'u2.base', 'ua.base'] 
    train_ratings,test_ratings,users,movies = dataset.load_ML100k(ml_100k_path)

    assert train_ratings.shape == (90000, 3)
    assert test_ratings.shape == (10000, 3)
    assert users.shape == (943, 5)
    assert movies.shape == (1682, 24)

def main(data_dir):
    
    # ML100K tests
    ml_100k_path=f'{data_dir}ml-100k/raw/ml-100k/'
    test_ML100K(ml_100k_path)

if __name__ == '__main__':
    data_dir='./data/'
    main(data_dir)
