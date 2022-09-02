from typing import Sequence, Tuple
import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import DeepLift,IntegratedGradients, NoiseTunnel, FeatureAblation, GradientShap
from torch.utils.data import Dataset, DataLoader


class BayesianNetwork(torch.nn.Module):

    def __init__(self, in_features, layers: Sequence[int], data_noise: float, dropout_prob: float = None):
        super(BayesianNetwork, self).__init__()
        if len(layers) < 1:
            raise ValueError("Network needs at least an output layer")

        self.dropout_prob = dropout_prob
        self.data_noise = data_noise
        self.layers = layers
        # Create linear stack
        # TODO: Support different activation functions
        self.stack = torch.nn.Sequential()
        for i in range(len(layers)):
            if i == 0:
                self.stack.append(torch.nn.Linear(in_features, layers[i]))
            elif i == (len(layers) - 1):
                self.stack.append(torch.nn.Linear(layers[i - 1], layers[i], bias=False))
            else:
                self.stack.append(torch.nn.Linear(layers[i - 1], layers[i]))
            self.stack.append(torch.nn.ReLU())
            if self.dropout_prob is not None:
                self.stack.append(torch.nn.Dropout(dropout_prob))

        self.__init_variances = torch.Tensor(len(layers))
        self.__lambdas = torch.Tensor(len(layers))
        self.__init_weights = []
        self.__init_biases = []
        self.initialise_from_prior()
        # print("init")

    def initialise_from_prior(self):
        # Even layers are the linear layers
        # self.__init_variances = [1.0 / layer.in_features for i, layer in enumerate(self.stack) if i % 2 == 0] # Tune 1.0
        # self.__lambdas = self.__init_variances / self.data_noise
        for i, layer in enumerate(self.layers):
            linear_idx = i * 2 if self.dropout_prob is None else i * 3
            mean_biases = torch.zeros(layer)
            mean_weights = torch.zeros_like(self.stack[linear_idx].weight.data)
            self.__init_variances[i] = 1.0 / self.stack[linear_idx].in_features
            self.__lambdas[i] = self.data_noise / self.__init_variances[i]
            std = torch.sqrt(self.__init_variances[i])
            self.__init_weights.append(torch.normal(mean_weights, std))
            self.stack[linear_idx].weight.data = self.__init_weights[-1]

            if i != (len(self.layers) - 1):
                if i == 0:
                    self.__init_biases.append(torch.normal(mean_biases, torch.sqrt(self.__init_variances[i] * self.stack[linear_idx].in_features)))
                else:
                    self.__init_biases.append(torch.normal(mean_biases, std))
                self.stack[linear_idx].bias.data = self.__init_biases[-1]

    def regularization(self):
        """Returns the total regularization term, before normalisation by the number of samples"""
        loss_unnorm = 0
        for i in range(len(self.layers)):
            linear_idx = i * 2 if self.dropout_prob is None else i * 3
            loss_unnorm += torch.sum(torch.square(self.stack[linear_idx].weight.data - self.__init_weights[i])) * self.__lambdas[i]
            if i != (len(self.layers) - 1):
                loss_unnorm += torch.sum(torch.square(self.stack[linear_idx].bias.data - self.__init_biases[i])) * self.__lambdas[i]

        return loss_unnorm

    def importance(self, train, test, norm: bool = True):
        is_training_at_start = self.training
        if self.training:
            self.eval()

        dl = DeepLift(self)
        dl_attr_test = dl.attribute(test)
        dl_sum = dl_attr_test.detach().sum(dim=0)
        dl_norm_sum = dl_sum / torch.linalg.norm(dl_sum, ord=1)
        if is_training_at_start:
            self.train()
        if norm:
            return dl_norm_sum
        else:
            return dl_sum

    def forward(self, x):
        return self.stack(x)


class BayesianNetworkEnsemble:

    def __init__(self, in_features: int, layers: Sequence[int], noise_variance, dropout_prob: float = None, no_models: int = 10):
        self.models = [BayesianNetwork(in_features, layers, noise_variance, dropout_prob) for _ in range(no_models)]
        self.in_features = in_features
        self.layers = layers
        self.noise_variance = noise_variance
        self.no_models = no_models

    def __getitem__(self, item) -> BayesianNetwork:
        return self.models[item]

    def __len__(self):
        return len(self.models)

    def __call__(self, input: torch.Tensor, return_predictions: bool = False):
        return self.predict(input, return_predictions)

    def predict(self, input: torch.Tensor, return_predictions: bool = False):
        with torch.no_grad():
            preds = torch.empty((self.no_models, len(input), 1))
            for i, m in enumerate(self.models):
                preds[i] = m(input)

            pred_mean = preds.mean(dim=0)
            pred_std = preds.std(dim=0)
            if return_predictions:
                return pred_mean, pred_std, preds
            else:
                return pred_mean, pred_std

    def importance(self, train, test, relative: bool = False, return_std: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        importances = torch.empty((self.no_models, self.in_features))
        for i in range(self.no_models):
            importances[i] = self.models[i].importance(train, test, norm=False)
        avg = importances.mean(dim=0)
        std = importances.std(dim=0)
        if relative:
            norm = torch.linalg.norm(avg, ord=1)
            avg /= norm
            std /= norm
        if return_std:
            return avg, std
        else:
            return avg

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()


def train_loop(features, labels, model, loss_fn, opt, batchsize, verbose=False):
    if len(features) != len(labels):
        raise ValueError("Features and labels must have same length")
    size = len(features)
    log = []
    no_batches = size // batchsize + 1
    rand_indices = torch.randperm(size)
    for i in range(no_batches):
        batch_start = i * batchsize
        batch_end = min((i + 1) * batchsize,size)
        rand_indices = torch.randperm(size)
        if batch_end != batch_start:
            X = features[rand_indices[batch_start:batch_end]]
            y = labels[rand_indices[batch_start:batch_end]]
            pred = model(X)
            loss = loss_fn(pred, y)
            loss += model.regularization() / size

            # Backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 10 == 0:
                loss, current = loss.item(), i * len(X)
                if math.isnan(loss):
                    raise RuntimeError("Loss shouldn't be nan")
                if verbose:
                    print(f"\rloss: {loss:>7f} [{current:>5d}/{size:>5d}]", end="")
                log.append(loss)

    return log

def log_likelihood(pred_mean, pred_std, target):
    log_2pi = 1.83787706641
    return torch.mean(0.5 * (torch.square(pred_mean - target) / torch.square(pred_std) + torch.log(
        torch.square(pred_std)) + log_2pi))


def ensemble_predict(models, X):
    with torch.no_grad():
        preds = torch.stack([m(X) for m in models])
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std

if __name__== "__main__":

    #    model = BayesianNetwork(6, data_noise=0, layers=[3, 2, 1])
    import warnings
    warnings.filterwarnings("error")



    noise = 0.1
    x_train = torch.linspace(-1, 1, 10000)
    x_train = x_train.reshape(-1, 1)
    y_gt = 0.5 * torch.sin(x_train * 5 * torch.pi + 5 * torch.pi) + 0.5
    y_train = y_gt + noise * torch.randn_like(y_gt)

    plt.plot(x_train, y_gt, c="r")
    plt.scatter(x_train, y_train, s=1)
    plt.show()

    # x_test = (torch.pi * torch.rand(100) + 9.5 * torch.pi) / (5 * torch.pi) - 1
    x_test = torch.rand(250) * 2.2 - 1
    x_test = x_test.sort()[0]
    y_test = 0.5 * torch.sin(x_test * 5 * torch.pi + 5 * torch.pi) + 0.5
    x_test = x_test.reshape(-1, 1)
    n_ensemble = 10

    models = [BayesianNetwork(1, layers=[50, 1], data_noise=noise ** 2) for _ in range(10)]

    # Sample prior
    with torch.no_grad():
        prior_mean, prior_std = ensemble_predict(models, x_test)

        plt.errorbar(y_test, prior_mean[:, 0], yerr=prior_std[:, 0], fmt="o")
        plt.plot(y_test, y_test, c="r", ls="-")
        plt.show()
        print("Yo")



    logs = []
    for i, m in enumerate(models):
        print(f"*****************************************************Training model #{i + 1} ")

        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(m.parameters(), lr=0.001)

        epochs = 200
        log = []
        for t in range(epochs):
            print(f"Epoch: #{t + 1:03}")
            curr_log = train_loop(x_train, y_train, m, loss_fn, optimiser, 32, verbose=True)
            log = log + curr_log
            print("")

        print("")
        logs.append(log)

    for log in logs:
        plt.plot(log)
    plt.show()

    post_mean, post_std = ensemble_predict(models, x_test)

    plt.errorbar(y_test, post_mean[:, 0], yerr=post_std[:, 0], fmt="o")
    plt.plot(y_test, y_test, c="r", ls="-")
    plt.show()
    upper, lower = post_mean + 2 * post_std, post_mean - 2 * post_std

    # torch.sort()
    plt.scatter(x_test[:, 0], y_test, label="Truth", c="C0")
    plt.plot(x_test[:, 0], post_mean[:, 0], label="Predict mean", c="C1")
    plt.fill_between(x_test[:, 0], upper[:, 0], lower[:, 0], label="95% confidence", alpha=0.4)
    plt.legend()
    plt.show()



