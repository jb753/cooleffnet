import argparse
import json
from pathlib import Path
import time
from datetime import datetime
from typing import List, Sequence, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch

from cooleffnet.fcdb import CoolingDatabase, Figure
from cooleffnet.util import CustomStandardScaler
from cooleffnet.bayesian_ensemble_network import BayesianNetworkEnsemble

RUN_ID = time.time()


def remove_minmax(array):
    mask = torch.logical_or(torch.eq(array, array.max()), torch.eq(array, array.min()))
    return torch.masked_select(array, mask=~mask)


def find_nearest_idx(array, val):
    return torch.abs(array - val).argmin()


def train_loop(features, labels, model, loss_fn, opt, batchsize, verbose=False, device="cpu"):
    if len(features) != len(labels):
        raise ValueError("Features and labels must have same length")
    size = len(features)
    log = []
    no_batches = size // batchsize + 1
    rand_indices = torch.randperm(size, device=device)
    for i in range(no_batches):
        batch_start = i * batchsize
        batch_end = min((i + 1) * batchsize, size)
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
                if verbose:
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                log.append(loss)

    return log


def train_ensemble(ensemble, epochs, training, test: tuple = None, verbose: bool = False, show_loss: bool = False,
                   device: str = "cpu", optargs: dict = None):
    training_feats, training_labels = training
    test_feats, test_labels = None, None
    if test is not None:
        test_feats, test_labels = test

    default_optargs = {
        'lr': 0.001,
        'conv_limit': 0.002,
        'batch_size': 256
    }

    if optargs is None:
        optargs = default_optargs
    else:
        for key, value in default_optargs.items():
            if key not in optargs or optargs[key] is None:
                optargs[key] = value

    convergence_limit = optargs['conv_limit']
    logs = [[] for _ in range(ensemble.no_models)]
    has_converged = [False] * ensemble.no_models
    converged_count = 0
    mean = -1
    stdev = -1
    ensemble.train()
    while not all(has_converged):
        for modelidx, model in enumerate(ensemble):
            if has_converged[modelidx]:
                continue
            else:
                model.initialise_from_prior()

            if verbose:
                print(f"Training model #{modelidx + 1}")
            loss_fn = torch.nn.MSELoss()
            optimiser = torch.optim.Adam(model.parameters(), lr=optargs['lr'], amsgrad=True)

            log = []
            for t in range(epochs):
                curr_log = train_loop(training_feats, training_labels, model, loss_fn, optimiser,
                                      batchsize=optargs['batch_size'], verbose=False, device=device)
                log = log + curr_log

            logs[modelidx] = log

        last_logs = torch.Tensor([log[-1] for log in logs])
        if mean == -1 or stdev == -1:
            mean = last_logs.mean().item()
            stdev = last_logs.std().item()
        has_converged = [x < convergence_limit and x < (mean + stdev) for x in last_logs]
        if has_converged.count(True) > converged_count:
            mean = last_logs.mean().item()
            stdev = last_logs.std().item()
            converged_count = has_converged.count(True)
        if verbose:
            print(f"{has_converged.count(True)}/{ensemble.no_models} converged"
                  f"{', retrying... ' if has_converged.count(True) != ensemble.no_models else ''}")

    if show_loss:
        for log in logs:
            offset = 10 if len(log) > 10 else 0
            plt.plot(log[offset:])
        plt.title("Loss over iterations")
        plt.show()

    with torch.no_grad():
        ensemble.eval()
        score = torch.nn.MSELoss()
        training_labels_pred, _ = ensemble(training_feats)
        train_score = torch.sqrt(score(training_labels_pred, training_labels))
        if test is not None:
            test_labels_pred, _ = ensemble(test_feats)
            test_score = torch.sqrt(score(test_labels_pred, test_labels))
            return train_score, test_score
        else:
            return train_score


def cross_validation(cv_training: Tuple[torch.Tensor, torch.Tensor],
                     cv_test: Tuple[torch.Tensor, torch.Tensor],
                     epochs: int,
                     layers: Sequence[int],
                     noise: float = 0.018,
                     no_models: int = 10,
                     optargs: dict = None,
                     stats: bool = True,
                     verbose: bool = False,
                     show_loss: bool = False,
                     show_importances: bool = False) -> List[BayesianNetworkEnsemble] | \
                                                        Tuple[List[BayesianNetworkEnsemble], Dict]:
    cv_training_feats, cv_training_labels = cv_training
    cv_test_feats, cv_test_labels = cv_test
    ensembles = []

    train_scores = torch.zeros([len(cv_training_feats)], dtype=torch.float)
    test_scores = torch.zeros([len(cv_training_feats)], dtype=torch.float)
    importances = torch.empty((len(cv_training_feats), cv_training_feats.shape[-1]))
    nodes = layers[0]
    for cv_idx in range(len(cv_training_feats)):

        if verbose:
            print(f"----- {nodes:03}/{cv_idx + 1:02} nodes -----")
        ensemble = BayesianNetworkEnsemble(cv_training_feats.shape[-1], layers=layers,
                                           noise_variance=noise ** 2, no_models=no_models)
        train_score, test_score = train_ensemble(ensemble, epochs,
                                                 (cv_training_feats[cv_idx], cv_training_labels[cv_idx]),
                                                 (cv_test_feats[cv_idx], cv_test_labels[cv_idx]), verbose=verbose,
                                                 show_loss=show_loss, optargs=optargs)
        train_scores[cv_idx] = train_score
        test_scores[cv_idx] = test_score
        ensembles.append(ensemble)
        importances[cv_idx] = ensemble.importance(cv_test_feats[cv_idx], relative=True)
        if verbose:
            print(f"Importances: {importances[cv_idx]}")

    avg_importances = importances.mean(dim=0)
    std_importances = importances.std(dim=0)
    if show_importances:
        plt.bar(avg_importances, yerr=std_importances)
        plt.show()

    masked_train = remove_minmax(train_scores)
    masked_test = remove_minmax(test_scores)
    if stats:
        stats = {
            'training_scores': train_scores.tolist(),
            'test_scores': test_scores.tolist(),
            'average_importances': avg_importances.tolist(),
            'stdev_importances': std_importances.tolist(),
            'training_average': train_scores.mean().item(),
            'masked_training_average': masked_train.mean().item(),
            'test_average': test_scores.mean().item(),
            'masked_test_average': masked_test.mean().item(),
            'training_stdev': train_scores.std().item(),
            'masked_training_stdev': masked_train.std().item(),
            'test_stdev': test_scores.std().item(),
            'masked_test_stdev': masked_test.std().item(),
        }
        return ensembles, stats
    else:
        return ensembles


def plot_true_predicted(true, predicted,
                        predicted_err: None,
                        title: str = None,
                        save_fig: bool = False,
                        show: bool = True):
    plt.figure(figsize=(16, 9))
    if predicted_err is not None:
        plt.errorbar(true, predicted, yerr=predicted_err, linewidth=0.5,
                     color='gray', ms=7, mfc='red', mec='black', fmt='o')
    else:
        plt.scatter(true, predicted, linewidth=0.5, color='gray', ms=7, mfc='red', mec='black')
    plt.plot(torch.arange(true.min(), true.max(), 0.01), torch.arange(true.min(), true.max(), 0.01),
             linewidth=3, linestyle='dashed', zorder=100)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    if title is not None:
        plt.title(title)
    if save_fig:
        unique_filename = ""
        if title is not None:
            unique_filename = "_" + title.lower().replace(" ", "_")
        plt.savefig(f"plots/true_vs_predicted{unique_filename}_{RUN_ID}.png")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train neural network on turbine film cooling database")
    parser.add_argument("-d", "--directory", type=str, help="Database directory", default=".")
    parser.add_argument("--loss", action="store_true", help="Show loss curves after training")
    parser.add_argument("--plot", action="store_true", help="Plot test set predictions at end")
    parser.add_argument("--prior", action="store_true", help="Show prior distribution")
    parser.add_argument("--export", action="store_true", help="Train on entire dataset and save network")
    parser.add_argument("-o", "--output", type=Path, help="Path to export model file")
    parser.add_argument("--holdout", action="store_true",
                        help="Train on a random subset of the dataset and test on the remaining")
    parser.add_argument("--training-min", type=int, help="Min. number of training examples", default=10000)
    parser.add_argument("--test-min", type=int, help="Min. number of test examples", default=2000)
    parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs to run", default=100)
    parser.add_argument("-n", "--nodes", type=int, help="Number of nodes in each hidden layer", default=100)
    parser.add_argument("--no-models", type=int, help="Number of models in ensemble", default=10)
    parser.add_argument("--noise", type=float, help="Magnitude of noise in data", default=0.018)
    parser.add_argument("--hidden", type=int, help="Number of hidden layers", default=1)
    parser.add_argument("--xnorm", type=str, help="Transform x/D some way", choices=["log", "reciprocal"])
    parser.add_argument("--conv-limit", type=float,
                        help="Maximum MSE loss a network can have to be considered converged")
    parser.add_argument("--batch-size", type=int, help="Batch size to use in training")
    parser.add_argument("--lr", type=float, help="Learning rate to be used in training")
    parser.add_argument("--cv", type=int, nargs='?', const=5,
                        help="Number of cross-validation sets")  # cv is None if not specified, 5 if cv given but no arg
    parser.add_argument("--comment", type=str, help="Comment to save in the run log", default="")
    parser.add_argument("--filter", type=str, help="Data filter", choices=["cylindrical", "shaped"], default=None)
    parser.add_argument("--params", type=str, help="Comma separated list of flow parameters",
                        default="AR,W/D,P/D,IR,BR,ER")

    args = parser.parse_args()
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    print(f"Using {device} device")

    db = CoolingDatabase(Path(args.directory))
    flow_params = args.params.split(",")
    feat_names = Figure.feature_names(list(map(Figure.to_param, flow_params)))
    epochs = args.epochs
    nodes = args.nodes
    no_hidden = args.hidden
    layers = [nodes] * no_hidden + [1]
    no_models = args.no_models
    noise = args.noise
    cv_count = args.cv
    training_min = args.training_min
    test_min = args.test_min
    data_filter = args.filter
    x_norm = args.xnorm if 'xnorm' in args else None

    print(f"Using variables {flow_params}\n"
          f"Running for {epochs} epochs, with network layers: {layers}\n")

    optargs = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'conv_limit': args.conv_limit
    }
    log_dict = {
        'id': RUN_ID,
        'date': datetime.utcfromtimestamp(int(RUN_ID)).strftime("%Y-%m-%d %H:%M:%S"),
        'training_minimum': training_min,
        'test_minimum': test_min,
        'example_count': db.get_example_count(),
        'all_files': sorted([f.name for f in db.get_all_files()]),
        'input_parameters': flow_params + ['x_D'],
        'filter': data_filter,
        'x_norm': x_norm,
        'no_nodes': nodes,
        'no_hidden': no_hidden,
        'epochs': epochs,
        'layers': layers,
        'no_models': no_models,
        'noise': noise,
        'comment': args.comment,
    }

    sc = CustomStandardScaler()
    if args.cv is not None:
        # Use entire dataset for cross validation
        db.generate_dataset(12000, 0, flow_param_list=flow_params, data_filter=data_filter, x_norm=x_norm)
        cv_training, cv_test = db.get_crossvalidation_sets(cv_count, padding="random")
        cv_training_feats, cv_training_labels = cv_training
        cv_test_feats, cv_test_labels = cv_test

        cv_training_feats = np.stack(cv_training_feats)
        cv_training_labels = np.stack(cv_training_labels)
        cv_test_feats = np.stack(cv_test_feats)
        cv_test_labels = np.stack(cv_test_labels)

        # Move everything to GPU if possible
        cv_training_feats = torch.from_numpy(cv_training_feats).float().to(device)
        cv_training_labels = torch.from_numpy(cv_training_labels).float().to(device)
        cv_test_feats = torch.from_numpy(cv_test_feats).float().to(device)
        cv_test_labels = torch.from_numpy(cv_test_labels).float().to(device)

        cv_training_feats = sc.fit_transform(cv_training_feats, dim=1)
        cv_test_feats = sc.transform(cv_test_feats)
        print(f"Running {cv_count}-fold cross-validation")

        cv_ensembles, cv_results = cross_validation((cv_training_feats, cv_training_labels),
                                                    (cv_test_feats, cv_test_labels),
                                                    epochs, layers, no_models=no_models, noise=noise,
                                                    stats=True, verbose=True, show_loss=args.loss, optargs=optargs)
        print(f"\r{nodes:03}/{len(cv_training_feats):02} nodes, "
              f"avg. training score: {cv_results['masked_training_average']:.3g} "
              f"σ = {cv_results['masked_training_stdev']:.3g}, "
              f"avg. test score: {cv_results['masked_test_average']:.3g} "
              f"σ = {cv_results['masked_test_stdev']:.3g}")
        print(f"All training scores: {cv_results['training_scores']}\nAll test scores: {cv_results['test_scores']}")
        print(f"Average feature importances: {dict(zip(feat_names,cv_results['average_importances']))}")
        print(f"Std. deviation of feature importances: {dict(zip(feat_names, cv_results['stdev_importances']))}")

        log_dict['cv_count'] = cv_count
        log_dict['cv_results'] = cv_results

    if args.holdout:
        # Use normal test-train split for plotting purposes
        db.generate_dataset(training_min, test_min, flow_param_list=flow_params, data_filter=data_filter, x_norm=x_norm)
        training_feats, training_labels = db.get_dataset()
        test_feats, test_labels = db.get_dataset(test=True)
        training_files = db.get_holdout_files()
        test_files = db.get_holdout_files(test=True)

        training_feats = sc.fit_transform(torch.from_numpy(training_feats).float().to(device), dim=0)
        training_labels = torch.from_numpy(training_labels).float().to(device)
        test_feats = sc.transform(torch.from_numpy(test_feats).float().to(device))
        test_labels = torch.from_numpy(test_labels).float().to(device)

        ensemble = BayesianNetworkEnsemble(len(training_feats[0]), layers, noise ** 2, no_models=no_models)

        if args.prior:
            with torch.no_grad():
                prior_mean, prior_std = ensemble(test_feats)
                plot_true_predicted(test_labels[:, 0], prior_mean[:, 0], predicted_err=prior_std[:, 0], title="Prior")

        print("Training on a subset of dataset and testing on a holdout")
        train_main_score, test_main_score = train_ensemble(ensemble, epochs, (training_feats, training_labels),
                                                           (test_feats, test_labels), verbose=False,
                                                           show_loss=args.loss, optargs=optargs)
        print(f"Training score: {train_main_score:.3g}, "
              f"Test score: {test_main_score:.3g}")

        score = torch.nn.MSELoss()
        if args.plot:
            with torch.no_grad():
                post_mean, post_std = ensemble(test_feats)
                plot_true_predicted(test_labels[:, 0], post_mean[:, 0], predicted_err=post_std[:, 0], title="Posterior")

            for file in test_files:
                f = Figure(file)
                feats, labels = f.get_feature_label_maps(flow_param_list=flow_params, x_norm=x_norm)
                feats_to_plot, _ = f.get_feature_label_maps(flow_param_list=flow_params, x_norm=None)
                study_name = file.name.split('_')[0]
                is_study_in_training = any(x.name.startswith(study_name) for x in training_files)
                corr_xs, corr_effs = f.get_correlations()
                fig, axes = plt.subplots(1, len(feats), figsize=(16, 9), sharey="all")

                fig.set_tight_layout(True)
                scores = []
                for feat, feat_plot, label, ax, corr_x, corr_eff in zip(feats, feats_to_plot, labels,
                                                                        np.atleast_1d(axes), corr_xs, corr_effs):
                    feat_torched = torch.from_numpy(feat).float()

                    feat_scaled = sc.transform(feat_torched)

                    label_pred_mean, label_pred_std = ensemble(feat_scaled)

                    upper, lower = label_pred_mean + 2 * label_pred_std, label_pred_mean - 2 * label_pred_std
                    ax.errorbar(feat_plot[:, -1], label, yerr=f.get_eff_uncertainty(), fmt="o", label="True value",
                                markersize=3)
                    ax.plot(feat_plot[:, -1], label_pred_mean[:, 0], color="orange", label="NN predicted")
                    ax.fill_between(feat_plot[:, -1], upper[:, 0], lower[:, 0], alpha=0.4, label="95% confidence")
                    ax.plot(corr_x, corr_eff, color="green", label="Correlation")
                    curr_score = torch.sqrt(score(torch.squeeze(label_pred_mean), torch.from_numpy(label).float()))
                    scores.append(curr_score)

                    ax.set_title(f"$\\bar{{\\Delta}}$: {curr_score.item():.3g}")
                    ax.set_ylim((0.0, 1.0))
                    # ax.legend()

                fig.suptitle(f"{f}, filter: {data_filter}\n"
                             f"Parameters used: {flow_params}, NN layers: {ensemble.layers}, "
                             f"x/D normalisation: {x_norm}\n"
                             f"{args.comment}\n"
                             f"Is same study in training set? {'Yes' if is_study_in_training else 'No'}\n"
                             f"Average MSE: {sum(scores) / len(scores):.3g}")

                plot_path = Path.cwd() / "plots" / (file.stem + f"_{RUN_ID}" + ".png")

                fig.savefig(plot_path)
                plt.show()

        log_dict['training_count'] = len(training_feats)
        log_dict['test_count'] = len(test_feats)
        log_dict['training_files'] = sorted([f.name for f in training_files])
        log_dict['test_files'] = sorted([f.name for f in test_files])
        log_dict['all_training_result'] = train_main_score.item()
        log_dict['all_test_result'] = test_main_score.item()

    if args.export:
        print("Training on all data in dataset to export & save model")
        db.generate_dataset(db.get_example_count(), 0, flow_param_list=flow_params,
                            data_filter=data_filter, x_norm=x_norm,)

        sc = CustomStandardScaler()
        full_training_feats, full_training_labels = db.get_dataset()
        full_training_feats = sc.fit_transform(torch.from_numpy(full_training_feats).float().to(device), dim=0)
        full_training_labels = torch.from_numpy(full_training_labels).float().to(device)

        full_ensemble = BayesianNetworkEnsemble(len(full_training_feats[0]), layers, noise ** 2, no_models=no_models)
        full_train_score = train_ensemble(full_ensemble, epochs, (full_training_feats, full_training_labels),
                                          verbose=True, show_loss=args.loss, optargs=optargs)

        print(f"Training loss root-MSE: {full_train_score}")

        export_path = Path(f"{int(RUN_ID)}.pth") if args.output is None else args.output

        export_dict = {
            "flow_params": flow_params,
            "x_norm": x_norm,
            "norm_mean": sc.mean,
            "norm_stdev": sc.std,
            "constructor": full_ensemble.get_constructor_params(),
            "state_dicts": full_ensemble.state_dicts()
        }
        torch.save(export_dict, export_path)

    log_path = Path.cwd() / "logs" / f"run_{RUN_ID}.json"
    with open(log_path, "w") as log_file:
        json.dump(log_dict, log_file, indent=2)


if __name__ == "__main__":
    main()
