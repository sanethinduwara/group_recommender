import torch
import numpy as np
import pandas as pd
from data_preprocessor import DataPreprocessor
from dataset import Dataset
from utility import Utility
from util.data_loader import DataLoader


class MovieRecommender(object):

    def __init__(self):
        #loading trained model
        self.model = torch.load("trained_model")
        self.dataset = Dataset()

    def get_recommendation_by_id(self, id, type="user"):
        return self._get_user_recommendations(id, k=10, r_type=type)

    def _get_user_recommendations(self, user_id, k=None, r_type="user"):

        _unrated_movies = None
        user_df = None
        member_df = None

        _utility = Utility(self.dataset)
        pre_processor = DataPreprocessor(self.dataset)

        # get movies which are not rated by user
        if r_type == "user":
            _unrated_movies = _utility.get_unrated_movies_by_user_id(user_id)
            users = self.dataset.load_users()
            user_df = users.loc[users['userId'] == user_id]
            user_df = pre_processor.transform_users_df(user_df)

        elif r_type == "group":
            _unrated_movies = _utility.get_unrated_movies_by_group_id(user_id)
            groups = self.dataset.load_groups()
            user_df = groups.loc[groups['groupId'] == user_id]
            if user_df.shape[0] == 0:
                groups = DataLoader().load_groups().iloc[:, 0:2]
                user_df = groups.loc[groups['groupId'] == user_id]
                user_df = pre_processor.transform_groups_df(user_df)

                users = pre_processor.transform_users_df()
                member_df = users[users.userId.isin(list(user_df.users)[0])]
                member_df.reset_index(drop=True, inplace=True)
            else:
                user_df = pre_processor.transform_groups_df(user_df)
                member_df = None

            user_df = user_df.drop(['users'], axis=1)

        user_df = pd.concat([user_df]*_unrated_movies.shape[0], ignore_index=True)
        _unrated_movies = pre_processor.transform_movies_df(_unrated_movies)

        user_df.reset_index(drop=True, inplace=True)
        _unrated_movies.reset_index(drop=True, inplace=True)

        data = pd.concat([user_df, _unrated_movies], axis=1)
        if r_type == "user":
            dataset_columns = list(data.columns)
            temp = dataset_columns[25]
            dataset_columns.remove(temp)
            dataset_columns.insert(1, temp)

            data = data[dataset_columns]

        _predicted_ratings = self.model(torch.Tensor(data.values), r_type, members=member_df)
        _predicted_ratings = _predicted_ratings * 5

        data['rating'] = torch.flatten(_predicted_ratings).tolist()

        data = data.drop(data.columns[2: len(data.columns)-1], axis=1)

        _top_recommendations = self._get_top_k_recommendations(data, k)
        return _top_recommendations

    def _get_top_k_recommendations(self, data, k=None):

        movie_index_mapping = np.load('movie_index_mapping.npy', allow_pickle='TRUE').item()

        # load all movies
        _all_movies = self.dataset.load_movies()

        data = data.sort_values(by=['rating'], ascending=False)
        top_k = data

        if k is not None:
            top_k = data[0:k]

        top_k.movieId = top_k.movieId.apply(lambda x: Utility.get_key_by_value(movie_index_mapping, x))
        recommendations = _all_movies[_all_movies.movieId.isin(top_k.movieId.tolist())]
        recommendations['rating'] = top_k.rating.tolist()
        return recommendations
