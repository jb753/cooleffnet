import argparse
import io
import csv
import json
from pathlib import Path
from typing import Sequence, Tuple

import torch
import matplotlib.pyplot as plt

from bayesian_ensemble_network import BayesianNetworkEnsemble
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
                clamp: bool = True) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] | torch.Tensor:
        """
        Predicts from a pretrained network
        Parameters
        ----------
        flow_params: torch.Tensor
            1D array of flow parameters required
        x_d_val: torch.Tensor
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
    def __transform_x(x: torch.Tensor, x_norm: str, norm_values: Sequence[float] = None, eps: float = 1e-7)\
            -> torch.Tensor:
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

    # TODO: Allow --info without requiring parameters
    parser.add_argument("model", type=Path,
                        help="Path to the pre-trained model")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument("--params", type=str,
                                help="Comma separated list of flow parameters as required by model", required=True)
    x_group = required_named.add_mutually_exclusive_group(required=True)
    x_group.add_argument("--x-vals", type=str, help="Comma separated list of x/D values where predictions are required")
    x_group.add_argument("--x-lin", type=str,
                         help="Comma separated arguments to torch.linspace() to generate x/D values\n"
                              "3 values expected: START,STOP,NUM")
    parser.add_argument("--norm-params", type=str,
                        help="Comma separated list of parameters required for normalising x/D values")
    parser.add_argument("--info", action="store_true",
                        help="Print information about the model in use")
    parser.add_argument("--plot", action="store_true",
                        help="Plot predictions")
    parser.add_argument("--format", type=str, choices=["json", "csv", "plain"], default="plain",
                        help="Format to output predictions")
    parser.add_argument("-o", "--output", type=Path,
                        help="Path to save output")
    parser.add_argument("--no-clip", action="store_false",
                        help="Don't clip predictions and confidence interval to [0;1]")
    parser.add_argument("--no-ci", action="store_false",
                        help="Don't include the 95%% confidence interval in the output")

    args = parser.parse_args()

    # TODO: Add default path or selection of models so everything can be distributed dogether
    pred = CoolingPredictor(args.model)

    if args.info:
        print(pred)
        exit(0)

    flow_params = torch.FloatTensor([float(s) for s in args.params.split(',')])
    norm_params = [float(s) for s in args.norm_params.split(',')] if args.norm_params is not None else None

    if args.x_vals is not None:
        x_vals = torch.FloatTensor([float(s) for s in args.x_vals.split(',')])
        if len(x_vals) < 1:
            raise ValueError(f"Expected at least one x/D value")
    elif args.x_lin is not None:
        linspace_params = args.x_lin.split(",")
        if len(linspace_params) != 3:
            raise ValueError(f"Expected 3 parameters for x-lin, got {len(linspace_params)} instead")

        x_vals = torch.linspace(float(linspace_params[0]), float(linspace_params[1]), int(linspace_params[2]))
    else:
        raise ValueError("One of x-vals or x-lin has to be specified. This should be enforced by argparse.")

    y_mean, (upper, lower) = pred(flow_params, x_vals, norm_params, clamp=args.no_clip)

    output_str = ""
    if args.format == "plain":
        output_str += f"{'x/D':10}{'Film eff.':10}"
        if args.no_ci:
            output_str += f"{'CI lower':10}{'CI upper':10}"
        output_str += "\n"
        output_str += "----------------------------------------\n"
        for x, eff, low, upp in zip(x_vals, y_mean, lower, upper):
            output_str += f"{x.item():<10.4f}{eff.item():<10.4f}"
            if args.no_ci:
                output_str += f"{low.item():<10.4f}{upp.item():<10.4f}"
            output_str += "\n"
    elif args.format == "csv":
        output_buffer = io.StringIO()
        writer = csv.writer(output_buffer)
        if args.no_ci:
            writer.writerow(["x/D", "Film effectiveness", "CI lower", "CI upper"])
        else:
            writer.writerow(["x/D", "Film. effectiveness"])

        for x, eff, low, upp in zip(x_vals, y_mean, lower, upper):
            if args.no_ci:
                writer.writerow([x.item(), eff.item(), low.item(), upp.item()])
            else:
                writer.writerow([x.item(), eff.item()])
        output_str = output_buffer.getvalue()
    elif args.format == "json":
        output_dict = {
            'x_d': x_vals.tolist(),
            'eff': y_mean.squeeze().tolist()
        }
        if args.no_ci:
            output_dict['ci_lower'] = lower.squeeze().tolist()
            output_dict['ci_upper'] = upper.squeeze().tolist()
        output_str = json.dumps(output_dict, indent=4)

    if args.output is not None:
        with open(args.output, mode="w") as output_file:
            print(output_str, file=output_file)
    else:
        print(output_str)

    if args.plot:
        plt.plot(x_vals, y_mean[:, 0], label="Prediction mean")
        plt.ylabel("Film effectiveness (-)")
        plt.xlabel("x/D (-)")
        plt.ylim((0.0, 1.0))
        plt.fill_between(x_vals, upper[:, 0], lower[:, 0], alpha=0.4, label="95% confidence interval")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
