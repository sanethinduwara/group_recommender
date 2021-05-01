import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from group_generator import GroupGenerator
from recommendation_test import RecommendationTest
from data_preprocessor import DataPreprocessor
from model.neural_net import NeuralNet
from cyclic_lr import CyclicLR
from model_evaluator import ModelEvaluator
from rating_iterator import RatingIterator
from dataset import Dataset
from service.group_service import GroupService
from utility import Utility
from sklearn.model_selection import train_test_split
from movie_recommender import MovieRecommender

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
from routes.group_api import group_api
from routes.user_api import user_api
from util.data_loader import DataLoader

RANDOM_STATE = 1
MODEL_STATS_FILE_PATH = "model_stats.npy"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
# app.register_blueprint(group_api, url_prefix='/api/group')
app.register_blueprint(user_api, url_prefix='/api/user')
cors = CORS(app, resources={r"/api/*": {"origins": "*"}}, headers="Content-Type")

group_service = GroupService()

def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)


def batches(X, y, bs=32, shuffle=True):
    for xb, yb in RatingIterator(X, y, bs, shuffle):
        xb = torch.Tensor(xb)
        yb = torch.FloatTensor(yb)
        yield xb, yb.view(-1, 1)


def train_model(net, dataset, optimizer, scheduler, stats, bs, r_type='user'):
    net.train()
    running_loss = 0
    loss_func = nn.MSELoss(reduction='sum')

    print("lr :", scheduler.get_lr())

    for batch in batches(*dataset, shuffle=False, bs=bs):
        x_batch, y_batch = [b.to(device) for b in batch]

        # calculate gradients
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            outputs = net(x_batch, r_type)
            loss = loss_func(outputs, y_batch)

            # update weights
            loss.backward()
            optimizer.step()
                
        running_loss += loss.item()

    scheduler.step()
    epoch_loss = running_loss / len(dataset[0])
    stats['train'] = epoch_loss
    
  
def validate_model(net, dataset, stats, bs, r_type='user'):
    net.eval()
    running_loss = 0

    loss_func = nn.MSELoss(reduction='sum')

    for batch in batches(*dataset, shuffle=False, bs=bs):
        x_batch, y_batch = [b.to(device) for b in batch]

        with torch.set_grad_enabled(False):
            outputs = net(x_batch, r_type)
            loss = loss_func(outputs, y_batch)
                
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataset[0])
    stats['val'] = epoch_loss


def initialize_service():
    lr = 1e-4
    wd = 1e-2
    bs = 256
    n_epochs = 25
    history = []

    set_random_seed(RANDOM_STATE)

    dataset = Dataset()
    data_preprocessor = DataPreprocessor(dataset)

    final_dataset = data_preprocessor.get_final_dataset()
    final_group_dataset = data_preprocessor.get_final_group_dataset()

    n_users, n_movies, n_groups = 10000, 5000, 15000
    print(
        f'Final Dataset: {final_dataset.userId.unique().size} users, {final_group_dataset[0].groupId.unique().size} groups, '
        f'{final_dataset.movieId.unique().size} movies')

    # split user train test dataset
    inputs_train, inputs_test, output_train, output_test = train_test_split(
        final_dataset.iloc[:, 0:44], final_dataset.rating, test_size=0.2, random_state=RANDOM_STATE)
    datasets = {'train': (inputs_train, output_train), 'val': (inputs_test, output_test)}
    dataset_sizes = {'train': len(inputs_train), 'val': len(inputs_test)}


    # split user train test dataset
    group_inputs_train, group_inputs_test, group_output_train, group_output_test = train_test_split(
        final_group_dataset[0].iloc[:, 0:20], final_group_dataset[0].rating, test_size=0.2, random_state=RANDOM_STATE)
    group_datasets = {'train': (group_inputs_train, group_output_train),
                      'val': (group_inputs_test, group_output_test)}

    # creating the neural network
    net = NeuralNet(n_users=n_users, n_movies=n_movies, n_groups=n_groups,
                    g_members=final_group_dataset[1],
                    n_factors=30, drop=0.2)

    net.to(device)
    # initiating an optimizer to reduce loss
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    iterations_per_epoch = dataset_sizes['train'] // bs
    # creating scheduler to reduce learning rate dynamically
    scheduler = CyclicLR(optimizer, CyclicLR.cosine(t_max=iterations_per_epoch * 2, eta_min=0))

    g_iterations_per_epoch = dataset_sizes['train'] // bs
    g_scheduler = CyclicLR(optimizer, CyclicLR.cosine(t_max=g_iterations_per_epoch * 2, eta_min=lr / 10))

    epochs, user_train_losses, user_val_losses, group_train_losses, group_val_losses = [], [], [], [], []

    for epoch in range(n_epochs):
        stats = {'epoch': epoch + 1, 'total': n_epochs}
        group_stats = {'epoch': epoch + 1, 'total': n_epochs}

        train_model(net, datasets['train'], optimizer, scheduler, stats, bs)
        train_model(net, group_datasets['train'], optimizer, g_scheduler, group_stats, bs, "group")

        validate_model(net, datasets['val'], stats, bs)
        validate_model(net, group_datasets['val'], group_stats, bs, "group")

        # history.append(stats)
        print('User : [{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
        print('Group : [{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**group_stats))
        epochs.append(epoch + 1)
        user_train_losses.append(stats['train'])
        user_val_losses.append(stats['val'])
        group_train_losses.append(group_stats['train'])
        group_val_losses.append(group_stats['val'])

    # calculate RMSE

    user_rmse = ModelEvaluator.calculate_RMSE(net, batches(*datasets['val'], shuffle=False, bs=bs))
    group_rmse = ModelEvaluator.calculate_RMSE(net, batches(*group_datasets['val'], shuffle=False, bs=bs), "group")

    print(f'Final User RMSE: {user_rmse:.4f}')
    print(f'Final Group RMSE: {group_rmse:.4f}')

    plt.plot(epochs, user_train_losses, label="user-train")
    plt.plot(epochs, user_val_losses, label="user-val")
    plt.plot(epochs, group_train_losses, label="group-train")
    plt.plot(epochs, group_val_losses, label="group-val")
    plt.xlabel('epoch')
    # naming the y axis
    plt.ylabel('loss')
    # giving a title to my graph
    plt.title('Loss variation with epoch')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()

    # saving the model
    torch.save(net, "trained_model")

    # creating a dictionary of model stats
    model_stats = {
        'user_rmse': user_rmse,
        'group_rmse': group_rmse,
        'user_train_losses': user_train_losses,
        'user_val_losses': user_val_losses,
        'group_train_losses': group_train_losses,
        'group_val_losses': group_val_losses
    }
    # saving model statistics to numpy array file
    np.save(MODEL_STATS_FILE_PATH, model_stats)


@app.route("/api/group", methods=['POST'])
def create_group():
    group_service.save_group(request.json)
    return jsonify(), 201


@app.route("/api/model/stats")
def model_statistics():
    if os.path.exists(MODEL_STATS_FILE_PATH):
        stats = np.load(MODEL_STATS_FILE_PATH, allow_pickle='TRUE').item()
    else:
        stats = {
            'user_rmse': 0,
            'group_rmse': 0,
            'user_train_losses': [],
            'user_val_losses': [],
            'group_train_losses': [],
            'group_val_losses': []
        }

    return jsonify(stats)


@app.route("/api/recommended/<id>")
def get_recommendations(id):
    movies_df = None
    # getting the value of type parameter
    r_type = request.args.get('type', default='user', type=str)
    # initializing recommender object
    recommender = MovieRecommender()

    # getting recommendations
    if r_type == 'group':
        movies_df = recommender.get_recommendation_by_id(int(id), type="group")
    else:
        movies_df = recommender.get_recommendation_by_id(int(id))
    # returning recommendations as JSON response
    return jsonify(movies_df.replace(np.nan, '', regex=True).to_dict('records')), 200


@app.route("/api/movies")
def get_all_movies():
    # returning movies as JSON response
    return jsonify(Dataset().load_movies().replace(np.nan, '', regex=True).to_dict('records')), 200


@app.route("/api/groups/user/<id>")
def get_groups_by_id(id):
    new_groups = Utility().get_groups_by_user_id(id, groups=DataLoader().load_groups())
    groups = Utility().get_groups_by_user_id(id)
    new_groups.reset_index(drop=True, inplace=True)
    groups.reset_index(drop=True, inplace=True)
    x = new_groups.append(groups)
    x.groupName = x.groupName.fillna('Unnamed')

    # returning groups as JSON response
    return jsonify(x.to_dict('records')), 200


if __name__ == "__main__":
    if not os.path.exists("trained_model"):
        initialize_service()
    # dataset = Dataset()
    # preprocessor = DataPreprocessor(dataset)
    # preprocessor.transform_users_df(write=True)
    # preprocessor.transform_movies_df(write=True)
    # preprocessor.transform_groups_df(write=True)
    # gg = GroupGenerator(preprocessor)
    #
    # ratings = dataset.load_ratings()
    # groups = gg.generate_groups(10, 10000)
    # r = gg.generate_group_ratings(ratings, groups, torch.load("trained_model"))


    # DataPreprocessor(Dataset()).get_final_dataset()
    # DataPreprocessor(Dataset()).get_final_group_dataset()
    # x = DataPreprocessor(Dataset()).run()

    app.run()


    #
    # RecommendationTest().accuracy2()
    # RecommendationTest().performance("group")
