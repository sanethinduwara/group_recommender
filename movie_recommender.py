import torch
import numpy as np
import pandas as pd
from dataset import Dataset
from utility import Utility
from rating_aggragator import RatingAggragator


class MovieRecommender(object):

    def __init__(self):
        #loading trained model
        self.model = torch.load("trained_model")
        self.dataset = Dataset()

    def get_recommendation_by_id(self, id, type="user"):
        return self._get_user_recommendations(id, k=10, r_type=type)
        # if type == "user":
        #     return self._get_user_recommendations(id, k=10, r_type=type)
        # if type == "group":
        #     return self._get_group_recommendations(id, k=10)

    def _get_user_recommendations(self, user_id, k=None, r_type="user"):

        _unrated_movies = None
        _utility = Utility(self.dataset.load_ratings())

        # get movies which are not rated by user
        if r_type == "user":
            _unrated_movies = _utility._get_unrated_movies_by_user_id(user_id)
        elif r_type == "group":
            _unrated_movies = _utility._get_unrated_movies_by_group_id(user_id)

        _unrated_movies_initial_ids = _unrated_movies[1].detach().clone()
        _unrated_movie_ids = _unrated_movies[1]
        _unrated_user_ids = _unrated_movies[0]

        movie_index_mapping = np.load('movie_index_mapping.npy', allow_pickle='TRUE').item()
        user_index_mapping = np.load('user_index_mapping.npy', allow_pickle='TRUE').item()

        _unrated_movie_ids = _unrated_movie_ids.apply_(lambda x: movie_index_mapping[x])

        if r_type == "user":
            _unrated_user_ids = _unrated_user_ids.apply_(lambda x: user_index_mapping[x])
        elif r_type == "group":
            _unrated_user_ids = _unrated_user_ids-1

        _predicted_ratings = self.model(_unrated_user_ids, _unrated_movie_ids, r_type)
        _predicted_ratings = _predicted_ratings * 5

        _top_recommendations = self._get_top_k_recommendations(
            _unrated_movies_initial_ids,
            _predicted_ratings.cpu().detach().numpy(),
            k
        )
        return _top_recommendations
            
        

    # def _get_group_recommendations(self, id, k=None):
    #     _group = self._get_group_by_id(id)
    #     _df_arr = []
    #     for _u_id in _group['userIds'][id-1].split(','):
    #         _user_ratings = self._get_user_recommendations(int(_u_id))
    #         _df_arr.append(_user_ratings)
            
    #     _df_arr = self._get_movies_in_common(_df_arr)
    #     _g_ratings = RatingAggragator.average(_df_arr)
    #     _g_ratings = _g_ratings.reset_index()
    #     if not k==None:
    #         return _g_ratings[0:k]
    #     return _g_ratings

    def _get_movies_in_common(self, df_arr):
        arr1 = df_arr[0]
        new_arr = []
        for x in range(1, len(df_arr)):
            arr1 = pd.merge(arr1, df_arr[x], how='inner', on=['movieId'])
        
        for y in range(len(df_arr)):
            new_arr.append(df_arr[y][df_arr[y]['movieId'].isin(list(arr1.movieId))]) 
        return new_arr

    def _get_top_k_recommendations(self, movies, ratings, k=None):

        # load all movies
        _all_movies = self.dataset.load_movies()

        _reshaped_prediction = np.vstack((movies, ratings.reshape(-1))).T
        _reshaped_prediction[:, 0].astype(np.int64)
        _sorted_ratings = _reshaped_prediction[np.argsort(-_reshaped_prediction[:, 1])]
        top_k = _sorted_ratings

        if not k==None:
            top_k = _sorted_ratings[0:k]

        top_k_as_df = pd.DataFrame({'movieId': top_k[:, 0], 'rating': top_k[:, 1]})
        return top_k_as_df.merge(_all_movies, on='movieId')

    def _get_group_by_id(self, id):
        _groups = self.dataset.load_generated_groups()
        return _groups.loc[_groups['groupId'] == id]
