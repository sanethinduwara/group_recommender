import pandas as pd


class DataLoader(object):

    __database_path = 'data/database/'

    def load_groups(self):
        group_cols = ['groupId', 'users', 'groupName']
        return pd.read_csv(self.__database_path + 'groups.csv', sep='\t', names=group_cols,
                           encoding='latin-1', skiprows=0)
