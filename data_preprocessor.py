import pandas as pd
import numpy as np


class DataPreprocessor:

    def __init__(self, dataset):
        self.dataset = dataset
    
    def get_transformed_groups(self):
        groups = self.dataset.load_groups()
        groups.users = groups.users.str.split(',').apply(lambda a: [int(x) - 1 for x in a])
        groups.groupId = groups.groupId-1
        return groups

    def transform_dataset(self, r_type="user"):
        ratings = None
        unique_users = None
        new_users = None
        user_to_index = None
        if r_type == "user":
            ratings = self.dataset.load_ratings()
            unique_users = ratings.userId.unique()

            user_to_index = {old: new for new, old in enumerate(unique_users)}
            new_users = ratings.userId.map(user_to_index)

        elif r_type == "group":
            ratings = self.dataset.load_group_ratings()
            unique_users = ratings.groupId.unique()

            user_to_index = {old: new for new, old in enumerate(unique_users)}
            new_users = ratings.groupId.map(user_to_index)

        unique_movies = ratings.movieId.unique()
        movie_to_index = {old: new for new, old in enumerate(unique_movies)}
        new_movies = ratings.movieId.map(movie_to_index)

        n_users = unique_users.shape[0]
        n_movies = unique_movies.shape[0]

        transformed_inputs = pd.DataFrame({r_type+'Id': new_users, 'movieId': new_movies})
        # normalize group ratings
        transformed_output = ratings['rating'].div(ratings['rating'].max()).astype(np.float32)

        # storing user/movie index mapping
        np.save('user_index_mapping.npy', user_to_index)
        np.save('movie_index_mapping.npy', movie_to_index)

        return (n_users, n_movies), (transformed_inputs, transformed_output)
