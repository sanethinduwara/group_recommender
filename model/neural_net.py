import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import zip_longest


class NeuralNet(nn.Module):
   
    def __init__(self, n_users, n_movies, n_factors=32, drop=0):
        
        super().__init__()
            
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.movie_embeddings = nn.Embedding(n_movies, n_factors)
       
       # prediction layers
        self.prediction_model = nn.Sequential(
            nn.Linear(3*n_factors, 8),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(8, 1),
            nn.Sigmoid()
            )

        # initial model
        for layer in self.modules():
            if isinstance(layer, nn.Embedding):
                nn.init.xavier_normal_(layer.weight)
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight)

        
    def forward(self, users, movies):

        u_embeddings = self.user_embeddings(users)
        m_embeddings = self.movie_embeddings(movies)

        element_embeddings = torch.mul(u_embeddings, m_embeddings)
        final_embedding = torch.cat((element_embeddings, u_embeddings, m_embeddings), dim=1)

        out = self.prediction_model(final_embedding)
        return out
    
   