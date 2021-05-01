import random
import numpy as np
import pandas as pd
import torch
from dataset import Dataset
from utility import Utility


class GroupGenerator(object):

    def __init__(self, data_preprocessor):
        self.data_preprocessor = data_preprocessor

    def group_users(self, users, n):
        random.shuffle(users)
        for i in range(0, len(users)-n, n):
            yield ",".join(str(x) for x in users[i:i + n])

    def generate_groups(self, n_members, n_groups):
        dataset = Dataset()
        generated_groups = []
        users = dataset.load_ratings().userId.unique()

        for i in range(n_groups):
            print("Generating Group...", "Group:", i)
            random.shuffle(users)
            n_members = random.randint(3, n_members)
            user_index_list = np.random.choice(len(users), n_members, replace=False)

            generated_groups.append([(i + 1), ",".join(str(users[x]) for x in user_index_list)])
        groups_df = pd.DataFrame(generated_groups, columns=['groupId', 'userIds'])
        groups_df.to_csv('data/generated/groups.csv', index=False, sep='\t')

        return groups_df

    def generate_group_ratings(self, ratings, groups, model):
        group_ratings = []

        movie_index_mapping = np.load('movie_index_mapping.npy', allow_pickle='TRUE').item()
        user_index_mapping = np.load('user_index_mapping.npy', allow_pickle='TRUE').item()

        movie_ids = ratings.movieId.map(movie_index_mapping).unique().astype(int)

        u_dataset = self.data_preprocessor.transform_users_df()
        m_dataset = self.data_preprocessor.transform_movies_df()

        for g_id, rows in groups.iterrows():
            print("Generating Group Rating...", "Group:", g_id)
            random_indices = np.random.choice((len(movie_ids)-1), 15, replace=False)
            user_ids = [user_index_mapping[k] for k in np.array(rows.userIds.split(',')).astype(int)]

            for idx in random_indices:
                filtered_users = u_dataset[u_dataset.userId.isin(user_ids)]
                filtered_movie = m_dataset[m_dataset.movieId == movie_ids[idx]]
                filtered_movies = pd.DataFrame(columns=filtered_movie.columns, dtype="int")

                for _ in range(filtered_users.shape[0]):
                    filtered_movies = pd.concat([filtered_movies, filtered_movie])

                filtered_users.reset_index(drop=True, inplace=True)
                filtered_movies.reset_index(drop=True, inplace=True)

                data = pd.concat([filtered_users, filtered_movies], axis=1)
                dataset_columns = list(data.columns)
                temp = dataset_columns[25]
                dataset_columns.remove(temp)
                dataset_columns.insert(1, temp)
                data = data[dataset_columns]
                ratings = model(
                    torch.Tensor(data.values),
                )
                group_ratings.append([
                    g_id + 1,
                    Utility.get_key_by_value(movie_index_mapping, movie_ids[idx]),
                    self.aggregate_ratings(torch.flatten(ratings*5).tolist())
                ])

        groups_df = pd.DataFrame(group_ratings, columns=['groupId', 'movieId', 'rating'])
        groups_df.to_csv('data/generated/group_ratings.csv', index=False, sep='\t')

        return group_ratings

    def aggregate_ratings(self, rating_list):
        return round(sum(rating_list)/len(rating_list))


