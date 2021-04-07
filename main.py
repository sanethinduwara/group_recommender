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

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


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

def train_model(net, dataset, optimizer, scheduler, criterion, stats, bs):
    running_loss = 0
    
    for batch in batches(*dataset, shuffle=True, bs=bs):
        x_batch, y_batch = [b.to(device) for b in batch]
        optimizer.zero_grad()

        # calculate gradients
        with torch.set_grad_enabled(True):
            outputs = net(x_batch[:,0], x_batch[:,1])
            loss = criterion(outputs, y_batch)
            
            # update weights
            scheduler.step()
            loss.backward()
            optimizer.step()
                
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataset[0])
    stats['train'] = epoch_loss
    
  
def test_model(net, dataset, optimizer, criterion,stats, bs):
    running_loss = 0
    optimizer.zero_grad()

    for batch in batches(*dataset, shuffle=False, bs=bs):
        x_batch, y_batch = [b.to(device) for b in batch]

        # calculate gradients
        with torch.set_grad_enabled(False):
            outputs = net(x_batch[:,0], x_batch[:,1])
            loss = criterion(outputs, y_batch)
                
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataset[0])
    stats['val'] = epoch_loss

@app.route("/api/retrain")
def retrain_model():
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

    (n, m), (X, y), _ = transform_dataset(ratings)
    print(f'Dataset: {n} users, {m} movies')

    # split train test dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    datasets = {'train': (x_train, y_train), 'val': (x_test, y_test)}
    dataset_sizes = {'train': len(x_train), 'val': len(x_test)}

    net = NeuralNet(n_users=n, n_movies=m, n_factors=32, drop=0.2)

    net.to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    iterations_per_epoch = dataset_sizes['train'] // bs
    scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))

    for epoch in range(n_epochs):
        stats = {'epoch': epoch + 1, 'total': n_epochs}

        train_model(net, datasets['train'], optimizer, scheduler, criterion, stats, bs)
        test_model(net, datasets['val'],optimizer, criterion, stats, bs)
        
        history.append(stats)
        print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))

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

    return jsonify({"RMSE": final_loss})

@app.route("/api/recommended/<id>")
def get_recommendations(id):

    r_type = request.args.get('type', default = 'user', type = str)

    # getting some recommendations
    recommender = MovieRecommender()
    movies_df = None

    if(r_type == 'group'):
        movies_df = recommender.get_recommendation_by_id(int(id), type="group")
    else:
        movies_df = recommender.get_recommendation_by_id(int(id))
    return jsonify(movies_df.replace(np.nan, '', regex=True).to_dict('records')), 200

@app.route("/api/movies")
def get_all_movies():
    dataset = Dataset()
    return jsonify(dataset.load_movies().replace(np.nan, '', regex=True).to_dict('records')), 200
    
@app.route("/api/groups/user/<id>")
def get_groups_by_id(id):
    util = Utility() 
    return jsonify(util.get_groups_by_user_id(id).to_dict('records')), 200

if __name__ == "__main__":
    app.run()