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
            nn.Linear(3 * n_factors, 8),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.attention = AttentionLayer(2 * n_factors, drop)

        # initialize model
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight)
            if isinstance(layer, nn.Embedding):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, users, movies, r_type='user'):

        if r_type == 'user':

            u_embeddings = self.user_embeddings(users)
            m_embeddings = self.movie_embeddings(movies)

            element_embeddings = torch.mul(u_embeddings, m_embeddings)
            final_embedding = torch.cat((element_embeddings, u_embeddings, m_embeddings), dim=1)

            out = self.prediction_model(final_embedding)
            return out

        elif r_type == 'group':
            groups = users
            group_embeds = torch.Tensor()
            item_embeds_full = self.movie_embeddings(torch.LongTensor(movies))
            for i, j in zip(groups, movies):
                members = self.group_members[i.item()]
                members_embeds = self.user_embeddings(torch.LongTensor(list(map(int, members))))
                items_numb = []
                for _ in members:
                    items_numb.append(j)
                item_embeds = self.movie_embeddings(torch.LongTensor(items_numb))
                group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
                at_wt = self.attention(group_item_embeds)
                g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
                group_embeds_pure = self.group_embeddings(torch.LongTensor([i]))
                g_embeds = g_embeds_with_attention + group_embeds_pure
                group_embeds = torch.cat((group_embeds, g_embeds))

            element_embeds = torch.mul(group_embeds, item_embeds_full)
            new_embeds = torch.cat((element_embeds, group_embeds, item_embeds_full), dim=1)
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
