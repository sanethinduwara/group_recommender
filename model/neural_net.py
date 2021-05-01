import torch
import torch.nn as nn


class NeuralNet(nn.Module):

    def __init__(self, n_users, n_groups, n_movies, g_members, n_factors=32, drop=0):

        super().__init__()

        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.movie_embeddings = nn.Embedding(n_movies, n_factors)
        self.group_embeddings = nn.Embedding(n_groups, n_factors)
        self.group_members = g_members

        # prediction layers
        self.prediction_model = nn.Sequential(
            nn.Linear(3 * n_factors + 42, 24),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(24, 1),
            nn.Sigmoid()
        )

        self.attention = AttentionLayer(2 * n_factors + 24, drop)

    def forward(self, data, r_type='user', members=None):

        if r_type == 'user':

            u_embeddings = self.user_embeddings(data[:, 0].type(torch.LongTensor))
            m_embeddings = self.movie_embeddings(data[:, 1].type(torch.LongTensor))

            features = data[:, 2:44].float()

            element_embeddings = torch.mul(u_embeddings, m_embeddings)
            final_embedding = torch.cat((element_embeddings, u_embeddings, m_embeddings, features), dim=1)

            out = self.prediction_model(final_embedding)
            return out

        elif r_type == 'group':
            group_embeds = torch.Tensor()
            all_weighted_features = torch.Tensor()
            item_embeds_full = self.movie_embeddings(data[:, 1].type(torch.LongTensor))
            # data[:, 0] - group ids, data[:, 1] - movie-ids
            for i, j in zip(data[:, 0], data[:, 1]):
                if members is None:
                    member_df = self.group_members[int(i.item())]
                else:
                    member_df = members
                members_embeds = self.user_embeddings(torch.LongTensor(member_df.iloc[:, 0]))
                movie_ids = torch.full((1, member_df.shape[0]), j)[0]

                item_embeds = self.movie_embeddings(movie_ids.type(torch.LongTensor))
                member_features = torch.Tensor(member_df.iloc[:, 1:25].values)
                gp_item_embeds_with_features = torch.cat((members_embeds, item_embeds, member_features), dim=1)
                at_wt = self.attention(gp_item_embeds_with_features)
                g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
                weighted_features = torch.matmul(at_wt, member_features)
                group_embeds_pure = self.group_embeddings(torch.LongTensor([i.type(torch.LongTensor)]))
                g_embeds = g_embeds_with_attention + group_embeds_pure
                group_embeds = torch.cat((group_embeds, g_embeds))
                all_weighted_features = torch.cat((all_weighted_features, weighted_features))

            all_features = torch.cat((all_weighted_features, data[:, 2:20]), dim=1)
            element_embeds = torch.mul(group_embeds, item_embeds_full)
            new_embeds = torch.cat((element_embeds, group_embeds, item_embeds_full, all_features), dim=1)
            return self.prediction_model(new_embeds)


class AttentionLayer(nn.Module):
    def __init__(self, n_factors, drop=0):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(n_factors, 16),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.sequential(x)
        weight = torch.softmax(out.view(1, -1), dim=1)
        return weight
