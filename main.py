import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from recommendation_test import RecommendationTest
from data_preprocessor import DataPreprocessor
from model.neural_net import NeuralNet
from cyclic_lr import CyclicLR
from model_evaluator import ModelEvaluator
from rating_iterator import RatingIterator
from dataset import Dataset
from utility import Utility
from sklearn.model_selection import train_test_split
from movie_recommender import MovieRecommender

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)


def batches(X, y, bs=32, shuffle=True):
    for xb, yb in RatingIterator(X, y, bs, shuffle):
        xb = torch.LongTensor(xb)
        yb = torch.FloatTensor(yb)
        yield xb, yb.view(-1, 1)


def train_model(net, dataset, optimizer, scheduler, loss_func, stats, bs, r_type='user'):
    running_loss = 0
    
    for batch in batches(*dataset, shuffle=True, bs=bs):
        x_batch, y_batch = [b.to(device) for b in batch]
        optimizer.zero_grad()

        # calculate gradients
        with torch.set_grad_enabled(True):
            outputs = net(x_batch[:, 0], x_batch[:, 1], r_type)
            loss = loss_func(outputs, y_batch)
            
            # update weights
            scheduler.step()
            loss.backward()
            optimizer.step()
                
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataset[0])
    stats['train'] = epoch_loss
    
  
def test_model(net, dataset, optimizer, loss_func, stats, bs, r_type='user'):
    running_loss = 0
    optimizer.zero_grad()

    for batch in batches(*dataset, shuffle=False, bs=bs):
        x_batch, y_batch = [b.to(device) for b in batch]

        # calculate gradients
        with torch.set_grad_enabled(False):
            outputs = net(x_batch[:, 0], x_batch[:, 1], r_type)
            loss = loss_func(outputs, y_batch)
                
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataset[0])
    stats['val'] = epoch_loss


@app.route("/api/retrain")
def retrain_model():
    RANDOM_STATE = 1
    lr = 1e-3
    wd = 1e-5
    bs = 256
    n_epochs = 11
    history = []

    set_random_seed(RANDOM_STATE)

    dataset = Dataset()
    data_preprocessor = DataPreprocessor(dataset)

    (n_users, n_movies), (inputs, output) = data_preprocessor.transform_dataset()
    print(f'Dataset: {n_users} users, {n_movies} movies')

    # split user train test dataset
    inputs_train, inputs_test, output_train, output_test = train_test_split(
        inputs, output, test_size=0.2, random_state=RANDOM_STATE)
    datasets = {'train': (inputs_train, output_train), 'val': (inputs_test, output_test)}
    dataset_sizes = {'train': len(inputs_train), 'val': len(inputs_test)}
    
    (n_groups, n_movies_g), (group_inputs, group_output) = data_preprocessor.transform_dataset("group")

    # split user train test dataset
    group_inputs_train, group_inputs_test, group_output_train, group_output_test = train_test_split(
        group_inputs, group_output, test_size=0.2, random_state=RANDOM_STATE)
    group_datasets = {'train': (group_inputs_train, group_output_train),
                      'val': (group_inputs_test, group_output_test)}

    net = NeuralNet(n_users=n_users, n_movies=n_movies, n_groups=n_groups,
                    g_members=data_preprocessor.get_transformed_groups().users,
                    n_factors=32, drop=0.2)

    net.to(device)


    loss_func = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    iterations_per_epoch = dataset_sizes['train'] // bs
    scheduler = CyclicLR(optimizer, CyclicLR.cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))

    g_iterations_per_epoch = dataset_sizes['train'] // bs
    g_scheduler = CyclicLR(optimizer, CyclicLR.cosine(t_max=g_iterations_per_epoch * 2, eta_min=lr/10))

    x1, s2, y1, y2 = [], [], [], []

    for epoch in range(n_epochs):
        stats = {'epoch': epoch + 1, 'total': n_epochs}
        group_stats = {'epoch': epoch + 1, 'total': n_epochs}

        train_model(net, datasets['train'], optimizer, scheduler, loss_func, stats, bs)
        train_model(net, group_datasets['train'], optimizer, g_scheduler, loss_func, group_stats, bs, 'group')

        test_model(net, datasets['val'], optimizer, loss_func, stats, bs)
        test_model(net, group_datasets['val'], optimizer, loss_func, group_stats, bs, 'group')
        
        # history.append(stats)
        print('User : [{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
        print('Group : [{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**group_stats))

    # calculate RMSE

        user_RMSE = ModelEvaluator.calculate_RMSE(net, batches(*datasets['val'], shuffle=False, bs=bs))
        group_RMSE = ModelEvaluator.calculate_RMSE(net, batches(*group_datasets['val'], shuffle=False, bs=bs), "group")

        print(f'Final User RMSE: {user_RMSE:.4f}')
        print(f'Final Group RMSE: {group_RMSE:.4f}')
        x1.append(epoch+1)
        y1.append(user_RMSE)
        y2.append(group_RMSE)



    plt.plot(x1, y1, label="user")
    plt.plot(x1, y2, label="group")
    plt.xlabel('epoch')
    # naming the y axis
    plt.ylabel('RMSE')
    # giving a title to my graph
    plt.title('RMSE variation with epoch')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()

    # saving the model
    torch.save(net, "trained_model")

    return jsonify({"RMSE": 'user_RMSE', "group_RMSE": 'group_RMSE'})


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
    # returning groups as JSON response
    return jsonify(Utility().get_groups_by_user_id(id).to_dict('records')), 200


if __name__ == "__main__":
    DataPreprocessor(Dataset()).transform_dataset()
    app.run()
