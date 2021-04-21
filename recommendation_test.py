import torch
import numpy as np
import pandas as pd
import time
from dataset import Dataset


class RecommendationTest(object):

    def __init__(self):
        # loading trained model
        self.model = torch.load("trained_model")
        self.dataset = Dataset()
        self.movie_index_mapping = np.load('movie_index_mapping.npy', allow_pickle='TRUE').item()
        self.user_index_mapping = np.load('user_index_mapping.npy', allow_pickle='TRUE').item()

    def accuracy(self):
        all_ratings = self.dataset.load_ratings()
        random_indices = np.random.choice(all_ratings.shape[0], 6, replace=False)
        df = pd.DataFrame()
        for i in random_indices:
            df = pd.concat([df, all_ratings[i:i+1]])
        print(df)



        predicted_ratings = self.model(
            torch.LongTensor(df.userId.values).apply_(lambda x: self.user_index_mapping[x]),
            torch.LongTensor(df.movieId.values).apply_(lambda x: self.movie_index_mapping[x])
        )
        predicted_ratings = predicted_ratings * 5
        print(predicted_ratings)

    def performance(self, r_type="user"):
        x = [50, 100, 200, 500, 1000, 2000, 3500]
        if r_type == "group":
            all_ratings = self.dataset.load_group_ratings()
        else:
            all_ratings = self.dataset.load_ratings()
        for n_prediction in x:
            df = all_ratings[0:n_prediction]
            movie_ids = torch.LongTensor(df.movieId.values).detach().clone()

            print('Time taken for', r_type, n_prediction, 'predictions :', end="\t")
            start_time = time.time()
            if r_type == "group":
                predicted_ratings = self.model(
                    torch.LongTensor(df.groupId.values),
                    movie_ids.apply_(lambda y: self.movie_index_mapping[y]),
                    r_type
                )
            else:
                user_ids = torch.LongTensor(df.userId.values).detach().clone()

                predicted_ratings = self.model(
                    user_ids.apply_(lambda y: self.user_index_mapping[y]),
                    movie_ids.apply_(lambda y: self.movie_index_mapping[y])
                )

            predicted_ratings = predicted_ratings * 5
            end_time = time.time()
            print((end_time - start_time)*1000, 'ms')
