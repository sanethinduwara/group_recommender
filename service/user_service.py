import pandas as pd


class UserService(object):

    def get_users(self):
        rating_cols = ['userId', 'gender', 'age', 'occupation', 'zipCode']
        return pd.read_table('data/ml-1m/users.dat', sep='\:\:', names=rating_cols, encoding='latin-1', skiprows=0)
