import argparse
from pathlib import Path
from typing import Sequence, Tuple

import torch
import matplotlib.pyplot as plt

from bayesian_ensemble_network import BayesianNetworkEnsemble
from fcdb import Figure
from util import CustomStandardScaler


class CoolingPredictor:

    def __init__(self, model_path: Path):
        self.__model_path = model_path

        import_dict = torch.load(self.__model_path)
        construct_params = import_dict['constructor']
        self.__x_norm = import_dict['x_norm']
        self.__sc = CustomStandardScaler()
        self.__sc.mean = import_dict['norm_mean']
        self.__sc.std = import_dict['norm_stdev']
        self.__flow_params = import_dict['flow_params']

        self.__ensemble = BayesianNetworkEnsemble(construct_params['in_features'], construct_params['layers'],
                                                construct_params['noise_variance'], construct_params['dropout_prob'],
                                                construct_params['no_models'])
        self.__ensemble.load_state_dicts(import_dict['state_dicts'])

    def info(self) -> str:
        """
        Information about the pretrained model
        Returns
        -------
        str
            Information string about the pretrained model
        """
        info_string = f"Model path: {self.__model_path}\n" \
                      f"Model params: {self.__ensemble.get_constructor_params()}\n" \
                      f"Expected flow parameters: {self.__flow_params}\n" \
                      f"X normalisation: {self.__x_norm}\n" \
                      f"Input normalisation mean: {self.__sc.mean}\n" \
                      f"Input normalisation std. dev: {self.__sc.std}\n"
        return info_string

    def predict(self,
                flow_params: torch.Tensor,
                x_d_val: torch.Tensor,
                norm_values: Sequence[float] = None,
                confidence: bool = True,
                clamp: bool = True) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predicts from a pretrained network
        Parameters
        ----------
        flow_params: torch.Tensor
            1D array of flow parameters required
        x_d_val: torch.tensor
            1D array of desired x/D values
        norm_values: list of float, optional
            If normalisation requires additional parameters (e.g. reciprocal normalisation)
        confidence: bool, default: True
            Include 95% confidence interval as a tuple of tensors referring to upper and lower boundary
        clamp : bool, default = True
            Clamp film effectiveness values and CI to the range of 0.0 and 1.0, as the ensemble might return values
            outside these boundaries


        Returns
        -------
        torch.Tensor
            Returns either a torch.Tensor with the mean prediction or a tuple of the mean prediction and a tuple of
            the upper and lower boundaries of the confidence interval
        """
        x_vals = x_d_val.clone().detach()
        if self.__x_norm is not None:
            x_vals = self.__transform_x(x_vals, self.__x_norm, norm_values)

        feats = torch.empty(len(x_vals), len(flow_params) + 1)
        feats[:, :-1] = flow_params.repeat(len(x_vals), 1)
        feats[:, -1] = x_vals

        feats = self.__sc.transform(feats)

        y_mean, y_std = self.__ensemble(feats)

        upper, lower = y_mean + 1.96 * y_std, y_mean - 1.96 * y_std
        if clamp:
            y_mean = torch.clamp(y_mean, 0.0, 1.0)
            upper = torch.clamp(upper, 0.0, 1.0)
            lower = torch.clamp(lower, 0.0, 1.0)
        if confidence:
            return y_mean, (upper, lower)
        else:
            return y_mean

    def __call__(self,
                 flow_params: torch.Tensor,
                 x_d_val: torch.Tensor,
                 norm_values: Sequence[float] = None,
                 confidence: bool = True,
                 clamp: bool = True) -> torch.Tensor:
        return self.predict(flow_params, x_d_val, norm_values)

    def __str__(self):
        return self.info()

    @staticmethod
    # TODO: Unify this and the one in fcdb.py or just move into a util.py
    def __transform_x(x: torch.Tensor, x_norm: str, norm_values: Sequence[float] = None, eps: float = 1e-7) -> torch.Tensor:
        """
        Utility function for transforming x/D in some way
        Parameters
        ----------
        x: torch.Tensor
            Input x/D values
        x_norm: { "log", "reciprocal" }
            Type of normalisation
        norm_values: list of float, optional
            Additional parameters required for normalisation (e.g. ACR and P/D for "reciprocal" normalisation")
        eps: float, default: 1e-7
            Small value added to x/D to avoid NaN values in both "log" and "reciprocal" normalisation

        Returns
        -------
        torch.Tensor
            The normalised x values, transformed according to x_norm
        """
        if x_norm == "log":
            return torch.log(x + eps)
        elif x_norm == "reciprocal":
            if norm_values is None:
                raise ValueError("Reciprocal normalisation selected but no ACR and P/D values given")
            if len(norm_values) != 2:
                raise ValueError(f"ACR and P/D values expected in norm_values, got {len(norm_values)} values  instead")
            return norm_values[0] / (norm_values[1] * (x + eps))
        else:
            raise ValueError(f"Unknown type of normalisation: {x_norm}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="Path to the pre-trained model")

    args = parser.parse_args()

    pred = CoolingPredictor(args.model)
    print(pred)

    f = Figure(Path("data/Saumweber2003_Fig8.json"))
    feats, labels = f.get_feature_label_maps(flow_param_list=['AR', 'W/P', 'P/D', 'BR', 'IR', 'ER'], x_norm="log")
    x_vals = torch.linspace(1, 3, 100)
    flow_params = torch.from_numpy(feats[0][0][:-1]).float()

    y_mean, (upper, lower) = pred(flow_params, x_vals, (feats[0][0][2], feats[0][0][5]))
    plt.plot(feats[0][:, -1], labels[0])
    plt.plot(x_vals, y_mean[:, 0])
    plt.fill_between(x_vals, upper[:, 0], lower[:, 0], alpha=0.4)
    plt.show()


if __name__ == "__main__":
    main()