import math
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.neural_net import NeuralNet
from cyclic_lr import CyclicLR
from rating_iterator import RatingIterator
from dataset import Dataset
from utility import Utility
from sklearn.model_selection import train_test_split
from group_generator import GroupGenerator
from movie_recommender import MovieRecommender

def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

def transform_dataset(ratings, top=None):
    if top is not None:
        ratings.groupby('userId')['rating'].count()
    
    unique_users = ratings.userId.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)
    
    unique_movies = ratings.movieId.unique()
    movie_to_index = {old: new for new, old in enumerate(unique_movies)}
    new_movies = ratings.movieId.map(movie_to_index)
    
    n_users = unique_users.shape[0]
    n_movies = unique_movies.shape[0]
    
    X = pd.DataFrame({'user_id': new_users, 'movie_id': new_movies})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_movies), (X, y), (user_to_index, movie_to_index)

def cosine(t_max, eta_min=0):

    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + math.cos(math.pi*t/t_max))/2
    
    return scheduler

def batches(X, y, bs=32, shuffle=True):
    for xb, yb in RatingIterator(X, y, bs, shuffle):
        xb = torch.LongTensor(xb)
        yb = torch.FloatTensor(yb)
        yield xb, yb.view(-1, 1)

if __name__ == '__main__':

    RANDOM_STATE = 1
    lr = 1e-3
    wd = 1e-5
    bs = 256
    n_epochs = 16
    patience = 10
    no_improvements = 0
    best_loss = np.inf
    best_weights = None
    history = []
    lr_history = []

    set_random_seed(RANDOM_STATE)

    dataset = Dataset()
    ratings = dataset.load_ratings()

    # normalize ratings
    ratings['rating'] = ratings['rating'].div(ratings['rating'].max())

    util = Utility(ratings)

    # gg = GroupGenerator()
    # groups = gg.generate_groups(ratings, 3)
    # g_ratings = gg.generate_group_ratings(ratings, groups)

    (n, m), (X, y), _ = transform_dataset(ratings)
    print(f'Dataset: {n} users, {m} movies')

    # split train test dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    datasets = {'train': (x_train, y_train), 'val': (x_test, y_test)}
    dataset_sizes = {'train': len(x_train), 'val': len(x_test)}

    net = NeuralNet(n_users=n, n_movies=m, n_factors=32, drop=0.2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net.to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    iterations_per_epoch = dataset_sizes['train'] // bs
    scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))

    for epoch in range(n_epochs):
        stats = {'epoch': epoch + 1, 'total': n_epochs}
        
        for phase in ('train', 'val'):
            if phase == 'train':
                training = True
            else:
                training = False

            running_loss = 0
            n_batches = 0
            
            for batch in batches(*datasets[phase], shuffle=training, bs=bs):
                x_batch, y_batch = [b.to(device) for b in batch]
                optimizer.zero_grad()

                
                # calculate gradients when training
                with torch.set_grad_enabled(training):
                    outputs = net(x_batch[:,0], x_batch[:,1])
                    loss = criterion(outputs, y_batch)
                    
                    # update weights only when training
                    if training:
                        scheduler.step()
                        loss.backward()
                        optimizer.step()
                        lr_history.extend(scheduler.get_lr())
                        
                running_loss += loss.item()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            stats[phase] = epoch_loss
            
            # early stopping before get overfitted
            if phase == 'val':
                if epoch_loss < best_loss:
                    print('loss improvement on epoch: %d' % (epoch + 1))
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(net.state_dict())
                    no_improvements = 0
                else:
                    no_improvements += 1
                    
        history.append(stats)
        print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
        if no_improvements >= patience:
            print('early stopping after epoch {epoch:03d}'.format(**stats))
            break

    # calculate RMSE

    expected_ratings, predictions = [], []

    with torch.no_grad():
        for batch in batches(*datasets['val'], shuffle=False, bs=bs):
            x_batch, y_batch = [b.to(device) for b in batch]
            outputs = net(x_batch[:, 0], x_batch[:, 1])
            expected_ratings.extend(y_batch.tolist())
            predictions.extend(outputs.tolist())

    expected_ratings = np.asarray(expected_ratings).ravel()
    predictions = np.asarray(predictions).ravel()

    final_loss = np.sqrt(np.mean((predictions - expected_ratings)**2))
    print(f'Final RMSE: {final_loss:.4f}')

    # saving the model
    torch.save(net, "trained_model")

    # getting some recommendations
    recommender = MovieRecommender()
    for x in range(10):
        print('---------------------', 'Recommendation for Group ', str(x+1), '---------------------')
        y = recommender.get_recommendation_by_id(x+1, type="group")
        print(y)
    
    for x in range(10):
        print('---------------------', 'Recommendation for User ', str(x+1), '---------------------')
        y = recommender.get_recommendation_by_id(x+1)
        print(y)
    