import pandas as pd


class Dataset(object):
    __dataset_path = 'data/ml-100k/'
    __generated_dataset_path = 'data/generated/'

    def __init__(self):
        self.movies = self.load_movies()
        self.user_ratings = self.load_ratings()

    def load_movies(self):
        movie_cols = ['movieId', 'movieTitle', 'releaseDate', 'videoReleaseDate',
                      'IMDbURL', 'unknown', 'Action', 'Adventure', 'Animation',
                      'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                      'FilmNoir', 'Horror', 'Musical', 'Mystery', 'Romance', 'SciFi',
                      'Thriller', 'War', 'Western']

        return pd.read_csv(self.__dataset_path + 'u.item', sep='|', names=movie_cols, encoding='latin-1')

    def load_ratings(self):
        rating_cols = ['userId', 'movieId', 'rating', 'timestamp']
        return pd.read_csv(self.__dataset_path + 'u.data', sep='\t', names=rating_cols, encoding='latin-1', skiprows=0)

    def load_groups(self):
        group_cols = ['groupId', 'users']
        return pd.read_csv(self.__generated_dataset_path + 'groups.csv', sep='\t', names=group_cols, encoding='latin-1')

    def load_group_ratings(self):
        group_rating_cols = ['groupId', 'movieId', 'rating']
        return pd.read_csv(self.__generated_dataset_path + 'group_ratings.csv', sep='\t', names=group_rating_cols,
                           encoding='latin-1')

    def full_dataset(self):
        return self.load_ratings().merge(self.load_movies(), on='movieId')

    def cleaned_data(self):
        return self.full_dataset().drop(['timestamp', 'movieTitle', 'releaseDate', 'videoReleaseDate', 'IMDbURL', 'unknown'], axis = 1)

    # load generated data
    def load_generated_groups(self):
        column_headers = ['groupId', 'userIds']
        return pd.read_csv(self.__generated_dataset_path + 'groups.csv', sep='\t', names=column_headers, encoding='latin-1', skiprows=1)
