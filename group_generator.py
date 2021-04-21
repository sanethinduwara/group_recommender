import pandas as pd
import random

from dataset import Dataset
from utility import Utility
import numpy as np
import torch


class GroupGenerator(object):

    def group_users(self, users, n):
        random.shuffle(users)
        for i in range(0, len(users)-n, n):
            yield ",".join(str(x) for x in users[i:i + n])

    def gen_random_groups(self, users, max_members, n_groups):
        group_list = []
        for i in range(n_groups):
            random.shuffle(users)
            n_members = random.randint(3, max_members)
            user_index_list = np.random.choice(len(users), n_members, replace=False)

            group_list.append([(i+1), ",".join(str(users[x]) for x in user_index_list)])

        return group_list

    def generate_groups(self, ratings, n_members, n_groups):
        dataset = Dataset()
        generated_groups = self.gen_random_groups(dataset.load_ratings().userId.unique(), n_members, n_groups)
        groups_df = pd.DataFrame(generated_groups, columns=['groupId', 'userIds'])
        groups_df.to_csv('data/generated/groups.csv', index=False, sep='\t')

        return groups_df

    def generate_group_ratings(self, ratings, groups, model):
        group_ratings = []

        movie_index_mapping = np.load('movie_index_mapping.npy', allow_pickle='TRUE').item()
        user_index_mapping = np.load('user_index_mapping.npy', allow_pickle='TRUE').item()

        movieIds = ratings.movieId.map(movie_index_mapping).unique().astype(int)

        for g_id, rows in groups.iterrows():
            random_indices = np.random.choice((len(movieIds)-1), 15, replace=False)
            user_ids = [user_index_mapping[k] for k in np.array(rows.userIds.split(',')).astype(int)]

            for idx in random_indices:
                movie_id_list = np.full((len(user_ids), 1), movieIds[idx], dtype=int)
                ratings = model(
                    torch.LongTensor(list(map(int, user_ids))),
                    torch.flatten(torch.LongTensor(movie_id_list))
                )
                group_ratings.append([
                    g_id + 1,
                    self._get_key_by_value(movie_index_mapping, movieIds[idx]),
                    self.aggregate_ratings(torch.flatten(ratings*5).tolist())
                ])

        groups_df = pd.DataFrame(group_ratings, columns=['groupId', 'movieId', 'rating'])
        groups_df.to_csv('data/generated/group_ratings.csv', index=False, sep='\t')

        return group_ratings

    def aggregate_ratings(self, rating_list):
        return round(sum(rating_list)/len(rating_list))

    def _get_key_by_value(self, dictionary, value):
        for key, val in dictionary.items():
            if value == val:
                return key
        return None
