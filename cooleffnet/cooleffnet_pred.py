import argparse
import io
import csv
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from cooleffnet.predictor import CoolingPredictor

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, nargs="?", default="general",
                        help="Name of pre-trained model or path to one. "
                             ""
                             "Pre-trained model options: \"general\", \"specific\"")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument("--params", type=str,
                                help="Comma separated list of flow parameters as required by model")
    x_group = required_named.add_mutually_exclusive_group()
    x_group.add_argument("--x-vals", type=str, help="Comma separated list of x/D values where predictions are required")
    x_group.add_argument("--x-lin", type=str,
                         help="Comma separated arguments to torch.linspace() to generate x/D values\n"
                              "3 values expected: START,STOP,NUM")
    parser.add_argument("--info", action="store_true",
                        help="Print information about the model in use")
    parser.add_argument("--norm-params", type=str,
                        help="Comma separated list of parameters if required for normalising x/D values")
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

    model = None
    if args.model in CoolingPredictor.get_pretrained_model_list():
        model = args.model
    else:
        model = Path(args.model)

    pred = CoolingPredictor(model)

    if args.info:
        print(pred)
        exit(0)
    else:
        if args.params is None:
            parser.error("flow parameters are required, specify them with --params")
        if args.x_vals is None and args.x_lin is None:
            parser.error("at least one of x-vals and x-lin need to specified")

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
