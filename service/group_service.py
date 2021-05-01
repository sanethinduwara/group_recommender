import pandas as pd
import numpy as np


class GroupService(object):

    def save_group(self, group):
        group_index_mapping = np.load('group_index_mapping.npy', allow_pickle='TRUE').item()
        group_id = max(group_index_mapping.keys()) + 1
        data = {'groupId': [group_id], 'users': [",".join(str(x) for x in group['users'])], 'groupName': [group['name']]}
        group_df = pd.DataFrame(data=data)
        group_index_mapping[group_id] = max(group_index_mapping.values()) + 1
        np.save('group_index_mapping.npy', group_index_mapping)

        group_df.to_csv('data/database/groups.csv', index=False, sep='\t', mode='a', header=False)
