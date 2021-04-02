import pandas as pd
import random
from utility import Utility
import numpy as np

class GroupGenerator(object):

    
    def group_users(self, users, n):
        random.shuffle(users)
        for i in range(0, len(users)-n, n):
            yield ",".join(str(x) for x in users[i:i + n])

    def generate_groups(self, ratings, n):
        arr = []
        util = Utility(ratings)
        g_ratings = util.group_by_movies()
        for ids in g_ratings['userId']:
            arr.extend(list(self.group_users(ids, n)))
        
        g_o = np.vstack((np.arange(1,len(arr)+1), np.array(arr))).T
        groups_df = pd.DataFrame(g_o, columns=['groupId', 'userIds'])
        groups_df.to_csv('data/generated/groups.csv', index=False, sep='\t')

        return groups_df


    def generate_group_ratings(self, ratings, groups):
        group_ratings = []
        movieIds = ratings.movieId.unique()
        for m_id in movieIds:
            for idx, u_ids in enumerate(groups.userIds[0:6]):
                rating_list = []
                for u_id in u_ids.split('|'):
                    filtered_rating = ratings[(ratings['userId']==u_id) & (ratings['movieId']==m_id)]
                    if (not filtered_rating.empty):
                        rating_list.append(filtered_rating['rating'])
                    else:
                        rating_list.append(0)
                agg_rating = self.aggregate_ratings(rating_list)
                group_ratings.append([idx+1, m_id, agg_rating])

        return group_ratings

    def aggregate_ratings(self, rating_list):
        return sum(rating_list)/len(rating_list)