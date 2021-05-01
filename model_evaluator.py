import torch
import numpy as np


class ModelEvaluator:

    @staticmethod
    def calculate_RMSE(net, batches, r_type="user"):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        expected_ratings, predictions = [], []

        with torch.no_grad():
            for batch in batches:
                x_batch, y_batch = [b.to(device) for b in batch]
                outputs = net(x_batch, r_type)
                expected_ratings.extend(y_batch.tolist())
                predictions.extend(outputs.tolist())

        expected_ratings = np.asarray(expected_ratings).ravel()
        predictions = np.asarray(predictions).ravel()

        final_loss = np.sqrt(np.mean((predictions - expected_ratings) ** 2))
        return final_loss
