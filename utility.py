import torch
import pandas as pd
import numpy as np
from dataset import Dataset


class Utility(object):

    def __init__(self, ratings=None):
        self.ratings = ratings

    def group_by_movies(self):
        filtered_ratings = self.ratings.groupby('movieId').filter(lambda x: x.userId.size>=5)
        return filtered_ratings.groupby('movieId')['userId'].apply(list).reset_index()

    def group_by_users(self):
        filtered_ratings = self.ratings.groupby('userId').filter(lambda x: x.movieId.size>=5)
        return filtered_ratings.groupby('userId')['movieId'].apply(list).reset_index()


    def _get_unrated_movies_by_user_id(self, user_id):
        grouped_u = self.group_by_users()
        m = grouped_u.loc[grouped_u['userId'] == user_id]
        all_movies = self.ratings['movieId'].unique()
        rated_movies = np.array(m['movieId'][user_id-1])
        unrated_movies = np.setdiff1d(all_movies, rated_movies)
        return torch.LongTensor(np.full((1, len(unrated_movies)), user_id)[0]), torch.LongTensor(unrated_movies)

    def get_groups_by_user_id(self, id):
        dataset = Dataset()
        groups = dataset.load_groups()
        return groups[pd.DataFrame(groups['users'].str.split(',').tolist()).isin([id]).any(1).values]
