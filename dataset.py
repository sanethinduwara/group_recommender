import pandas as pd


class Dataset(object):
    __dataset_path = 'data/ml-1m/'
    __generated_dataset_path = 'data/generated/'

    def load_movies(self):
        movie_cols = ['movieId', 'movieTitle', 'genre']
        return pd.read_csv(self.__dataset_path + 'movies.dat', sep='::', names=movie_cols, encoding='latin-1')

    def load_ratings(self):
        rating_cols = ['userId', 'movieId', 'rating', 'timestamp']
        return pd.read_table(self.__dataset_path + 'ratings.dat', sep='\:\:', names=rating_cols,
                             encoding='latin-1', skiprows=0)

    def load_users(self):
        rating_cols = ['userId', 'gender', 'age', 'occupation', 'zipCode']
        return pd.read_table(self.__dataset_path + 'users.dat', sep='\:\:', names=rating_cols,
                             encoding='latin-1', skiprows=0)

    def load_groups(self):
        group_cols = ['groupId', 'users']
        return pd.read_csv(self.__generated_dataset_path + 'groups.csv', sep='\t', names=group_cols,
                           encoding='latin-1', skiprows=1)

    def load_group_ratings(self):
        group_rating_cols = ['groupId', 'movieId', 'rating']
        return pd.read_csv(self.__generated_dataset_path + 'group_ratings.csv', sep='\t', names=group_rating_cols,
                           encoding='latin-1', skiprows=1)
