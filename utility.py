import pandas as pd
import numpy as np
from dataset import Dataset


class Utility(object):

    def __init__(self, dataset=None):
        self.dataset = dataset
        if dataset is not None:
            self.ratings = self.dataset.load_ratings()

    def group_by_movies(self):
        filtered_ratings = self.ratings.groupby('movieId').filter(lambda x: x.userId.size >= 5)
        return filtered_ratings.groupby('movieId')['userId'].apply(list).reset_index()

    def group_by_users(self):
        filtered_ratings = self.ratings.groupby('userId').filter(lambda x: x.movieId.size >= 5)
        return filtered_ratings.groupby('userId')['movieId'].apply(list).reset_index()

    def get_unrated_movies_by_user_id(self, user_id):
        movie_dataset = self.dataset.load_movies()
        grouped_u = self.group_by_users()
        m = grouped_u.loc[grouped_u['userId'] == user_id]
        all_movies = self.ratings['movieId'].unique()
        rated_movies = np.array(m.movieId.tolist()[0])
        unrated_movies = np.setdiff1d(all_movies, rated_movies)
        x = movie_dataset.loc[movie_dataset['movieId'].isin(np.setdiff1d(all_movies, rated_movies))]
        # return torch.LongTensor(np.full((1, len(unrated_movies)), user_id)[0]), torch.LongTensor(unrated_movies)
        return x

    def get_unrated_movies_by_group_id(self, group_id):
        movie_dataset = self.dataset.load_movies()
        return movie_dataset

    def get_groups_by_user_id(self, id, groups=None):
        if groups is None:
            dataset = Dataset()
            groups = dataset.load_groups()
        return groups[pd.DataFrame(groups['users'].str.split(',').tolist()).isin([id]).any(1).values]

    @staticmethod
    def get_key_by_value(dictionary, value):
        for key, val in dictionary.items():
            if value == val:
                return key
        return None
