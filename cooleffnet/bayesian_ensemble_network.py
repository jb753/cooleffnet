from typing import Sequence, Tuple, Dict, List, Union

import torch
from captum.attr import DeepLift


class BayesianNetwork(torch.nn.Module):

    def __init__(self,
                 in_features: int,
                 layers: Sequence[int],
                 data_noise: float,
                 dropout_prob: float = None):
        """
        A Bayesian neural network subclassed from torch.nn.Module
        Parameters
        ----------
        in_features: int
            Number of features
        layers: iterable of int
            Sizes of layers after the inputs (i.e. sizes of hidden layers and output layer)
        data_noise: float
            Noise variance of the input data
        dropout_prob: float, optional
            If given then Dropout layers are added with this dropout probability to the model
        """
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
        """Initialises weights and biases from a prior distribution, based on the implementation in
        `Bayesian Machine Learning for the Prognosis of Combustion Instabilities From Noise` by Sengupta et al.
        DOI: https://doi.org/10.1115/1.4049762"""
        for i, layer in enumerate(self.layers):
            # Linear layers are every second or every third depending on whether Dropout layers are included
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
                    self.__init_biases.append(
                        torch.normal(mean_biases,
                                     torch.sqrt(self.__init_variances[i] * self.stack[linear_idx].in_features)))
                else:
                    self.__init_biases.append(torch.normal(mean_biases, std))
                self.stack[linear_idx].bias.data = self.__init_biases[-1]

    def regularization(self) -> float:
        """Returns the total regularization term, before normalisation by the number of samples"""
        loss_unnorm = 0
        for i in range(len(self.layers)):
            linear_idx = i * 2 if self.dropout_prob is None else i * 3
            loss_unnorm += torch.sum(torch.square(self.stack[linear_idx].weight.data - self.__init_weights[i])) \
                * self.__lambdas[i]
            if i != (len(self.layers) - 1):
                loss_unnorm += torch.sum(torch.square(self.stack[linear_idx].bias.data - self.__init_biases[i])) \
                               * self.__lambdas[i]

        return loss_unnorm

    def importance(self, test, norm: bool = True) -> torch.Tensor:
        """
        Returns the importance of each input feature according the DeepLift feature attribution scheme
        Parameters
        ----------
        test : torch.Tensor
            Test input in a PyTorch Tensor
        norm: bool, optional
            If false, then no L1 normalisation is done
        Returns
        -------
        torch.Tensor
            Importance of each input feature in a PyTorch Tensor
        """
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

    def __init__(self,
                 in_features: int,
                 layers: Sequence[int],
                 noise_variance: float,
                 dropout_prob: float = None,
                 no_models: int = 10):
        """
        An ensemble of Bayesian neural networks, with convenient overloads for prediction.

        Parameters
        ----------
        in_features : int
            Number of features, see :py:class:`BayesianNetwork` for more details.
        layers : iterable of int
            Sizes of each layers, see :py:class:`BayesianNetwork` for more details.
        noise_variance : float
            Noise variance of input data, see :py:class:`BayesianNetwork` for more details.
        dropout_prob : float, optional
            Dropout probability of layers if not None, :py:class:`BayesianNetwork` for more details.
        no_models : int, default : 10
            Number of Bayesian neural networks to use in ensemble
        """
        self.models = [BayesianNetwork(in_features, layers, noise_variance, dropout_prob) for _ in range(no_models)]
        self.in_features = in_features
        self.layers = layers
        self.noise_variance = noise_variance
        self.dropout_prob = dropout_prob
        self.no_models = no_models

    def __getitem__(self, item) -> BayesianNetwork:
        return self.models[item]

    def __len__(self) -> int:
        return len(self.models)

    def __call__(self, input: torch.Tensor, return_predictions: bool = False) \
            -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.predict(input, return_predictions)

    # TODO: Add cutoff to keep mean and 95% confidence interval between 0 and 1
    def predict(self, input: torch.Tensor, return_predictions: bool = False)\
            -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Predicts
        Parameters
        ----------
        input : torch.Tensor
            Input features
        return_predictions: bool, optional
            If True then all individial predictions are returned as well
        Returns
        -------
        tuple
            Tuple of mean and standard deviation of predictions and all individal predictions if
        """
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

    def importance(self, test: torch.Tensor, relative: bool = False, return_std: bool = False) \
            -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns average importance of input features
        Parameters
        ----------
        test: torch.Tensor
            Test input
        relative: bool, optional
            If True normalise by L1 norm
        return_std: bool, optional
            If True returns standard deviation of each feature importance

        Returns
        -------
        torch.Tensor
            Average importance of each feature

        """
        importances = torch.empty((self.no_models, self.in_features))
        for i in range(self.no_models):
            importances[i] = self.models[i].importance(test, norm=False)
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
        """Sets all models in the ensemble to training mode"""
        for model in self.models:
            model.train()

    def eval(self):
        """Sets all models in the ensemble to evaluation mode"""
        for model in self.models:
            model.eval()

    def get_constructor_params(self):
        """
        Returns all parameters necessary to construct an empty ensemble with the same setup
        Returns
        -------
        dict
            Dictionary of all parameters necessary to construct an empty ensemble with the same structure.
        """
        constructor_params = {
            "no_models": len(self.models),
            "in_features": self.in_features,
            "layers": self.layers,
            "noise_variance": self.noise_variance,
            "dropout_prob": self.dropout_prob,
        }
        return constructor_params

    def state_dicts(self) -> List[torch.Tensor]:
        """
        Returns a list of all the state dicts in the network
        Returns
        -------
        list
            Returns a list of all the state dicts in the network
        """
        return [m.state_dict() for m in self.models]

    def load_state_dicts(self, state_dicts: Sequence[Dict]):
        """
        Loads state dictionaries into models
        Parameters
        ----------
        state_dicts
            List of state dicts
        Returns
        -------

        """
        if len(self.models) != len(state_dicts):
            raise ValueError("Number of models in state dict and number of models in network")
        for m, state_dict in zip(self.models, state_dicts):
            m.load_state_dict(state_dict)


def log_likelihood(pred_mean, pred_std, target):
    """
    Calculated log-likelihood of a prediction
    Parameters
    ----------
    pred_mean : torch.Tensor
        Mean of predictions
    pred_std : torch.Tensor
        Standard deviation of predictions
    target : torch.Tensor
        True value

    Returns
    -------
    float
        Log-likelihood of prediction
    """

    log_2pi = 1.83787706641
    return torch.mean(0.5 * (torch.square(pred_mean - target) / torch.square(pred_std) + torch.log(
        torch.square(pred_std)) + log_2pi))
