import numpy as np
import pandas as pd


class DataPreprocessor:

    def __init__(self, dataset):
        self.dataset = dataset

    def preprocess_rating_dataset(self):

        user_index_mapping = np.load('user_index_mapping.npy', allow_pickle='TRUE').item()
        movie_index_mapping = np.load('movie_index_mapping.npy', allow_pickle='TRUE').item()

        ratings = self.dataset.load_ratings()
        ratings.userId = ratings.userId.apply(lambda x: user_index_mapping[x])
        ratings.movieId = ratings.movieId.apply(lambda x: movie_index_mapping[x])
        ratings.rating = ratings.rating/ratings.rating.max()
        ratings = ratings.iloc[:, 0:3]

        return ratings

    def transform_users_df(self, df=None, write=False):
        user_index_mapping = np.load('user_index_mapping.npy', allow_pickle='TRUE').item()

        if df is None:
            df = self.dataset.load_users()

        if write:
            self.__column_reindex(df, "userId", "user_index_mapping.npy")
        else:
            df.userId = df.userId.apply(lambda x: user_index_mapping[x])
        df.age = round((df.age / df.age.max()), 4)

        df.zipCode = df.zipCode.apply(lambda x: x.split('-')[0])
        df.zipCode = pd.to_numeric(df.zipCode)
        df.zipCode = round((df.zipCode / df.zipCode.max()), 4)

        df = self.__hot_encode_occupation(df)
        df = df.drop(['occupation'], axis=1)
        df.gender = df.gender.apply(lambda x: 1 if x == "M" else 0)

        return df

    def transform_movies_df(self, df=None, write=False):
        movie_index_mapping = np.load('movie_index_mapping.npy', allow_pickle='TRUE').item()

        if df is None:
            df = self.dataset.load_movies()

        if write:
            self.__column_reindex(df, "movieId", "movie_index_mapping.npy")
        else:
            df.movieId = df.movieId.apply(lambda x: movie_index_mapping[int(x)])

        df = self.__hot_encode_genres(df)
        df = df.drop(['movieTitle', 'genre'], axis=1)
        return df

    def transform_groups_df(self, df=None, write=False):
        user_index_mapping = np.load('user_index_mapping.npy', allow_pickle='TRUE').item()
        group_index_mapping = np.load('group_index_mapping.npy', allow_pickle='TRUE').item()

        if df is None:
            df = self.dataset.load_groups()

        if write:
            self.__column_reindex(df, "groupId", "group_index_mapping.npy")
        else:
            df.groupId = df.groupId.apply(lambda x: group_index_mapping[x])

        df.users = df.users.apply(lambda a: [user_index_mapping[int(y)] for y in a.split(",")])

        return df

    def transform_group_rating_dataset(self):

        group_index_mapping = np.load('group_index_mapping.npy', allow_pickle='TRUE').item()
        movie_index_mapping = np.load('movie_index_mapping.npy', allow_pickle='TRUE').item()

        ratings = self.dataset.load_group_ratings()
        ratings.groupId = ratings.groupId.apply(lambda x: group_index_mapping[x])
        ratings.movieId = ratings.movieId.apply(lambda x: movie_index_mapping[x])

        ratings.rating = ratings.rating/ratings.rating.max()

        return ratings

    def __column_reindex(self, dataset, column_name, file_name):
        unique_items = dataset[column_name].unique()
        item_to_index = {old: new for new, old in enumerate(unique_items)}
        new_items = dataset[column_name].map(item_to_index)
        np.save(file_name, item_to_index)
        dataset[column_name] = new_items

    def __hot_encode_genres(self, dataset):
        genres = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']

        def encode(genre_list):
            encoded_genres = []
            for genre in genres:
                encoded_genres.append(1 if genre in genre_list else 0)
            return encoded_genres

        encoded_series = dataset.genre.apply(lambda x: encode(x.split('|')))
        genre_df = pd.DataFrame(list(encoded_series), columns=genres)

        dataset.reset_index(drop=True, inplace=True)
        genre_df.reset_index(drop=True, inplace=True)

        return pd.concat([dataset, genre_df], axis=1)

    def __hot_encode_occupation(self, dataset):
        occupations = ["other", "academic/educator", "artist", "clerical/admin", "college/grad student",
                      "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker",
                      "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist",
                      "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]

        def encode(occupation):
            encoded_occupations = []
            for occ in occupations:
                encoded_occupations.append(1 if occ == occupations[occupation] else 0)
            return encoded_occupations

        encoded_series = dataset.occupation.apply(lambda x: encode(x))
        occupation_df = pd.DataFrame(list(encoded_series), columns=occupations)
        dataset.reset_index(drop=True, inplace=True)
        return pd.concat([dataset, occupation_df], axis=1)

    def get_final_dataset(self):
        transformed_users = self.transform_users_df(write=True)
        transformed_movies = self.transform_movies_df(write=True)
        transformed_ratings = self.preprocess_rating_dataset()
        transformed_ratings = transformed_ratings.merge(transformed_users, on='userId')
        transformed_ratings = transformed_ratings.merge(transformed_movies, on='movieId')

        # moving rating column to the end
        column_names = list(transformed_ratings.columns)
        temp = column_names[2]
        column_names.remove(temp)
        column_names.append(temp)

        return transformed_ratings[column_names]

    def get_final_group_dataset(self):
        transformed_groups = self.transform_groups_df(write=True)
        transformed_movies = self.transform_movies_df()
        transformed_users = self.transform_users_df()
        transformed_group_rating = self.transform_group_rating_dataset()
        transformed_group_rating = transformed_group_rating.merge(transformed_groups, on='groupId')
        transformed_group_rating = transformed_group_rating.merge(transformed_movies, on='movieId')

        transformed_group_rating = transformed_group_rating.drop(['users'], axis=1)

        # moving ratings column to the end
        column_names = list(transformed_group_rating.columns)
        temp = column_names[2]
        column_names.remove(temp)
        column_names.append(temp)

        # create dictionary with member dataframes against group id
        dic = {}
        for index, row in transformed_groups.iterrows():
            member_df = transformed_users[transformed_users.userId.isin(row.users)]
            member_df.reset_index(drop=True, inplace=True)
            dic[row.groupId] = member_df

        return transformed_group_rating[column_names], dic

