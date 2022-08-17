import argparse
import json
from pathlib import Path
import time
from datetime import datetime
from typing import Sequence, Tuple, Dict, Union

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from fcdb import CoolingDatabase, Figure
from util import CustomMinMaxScaler, CustomStandardScaler
from bayesian_ensemble_network import BayesianNetworkEnsemble

RUN_ID = int(time.time())
def remove_minmax(array):
    mask = torch.logical_or(torch.eq(array, array.max()), torch.eq(array,array.min()))
    #a_masked = np.ma.masked_array(array, mask=mask)
    return torch.masked_select(array, mask=~mask)


def find_nearest_idx(array, val):
    return torch.abs(array - val).argmin()

class CoolingDataset(Dataset):
    def __init__(self, feats, labels, transform=None, target_transform=None):
        self.feats = feats
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        feat = self.feats[idx]
        label = self.labels[idx]
        if self.transform is not None:
            feat = self.transform(feat)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return feat, label


def train_loop(features, labels, model, loss_fn, opt, batchsize, verbose=False, device="cpu"):
    if len(features) != len(labels):
        raise ValueError("Features and labels must have same length")
    size = len(features)
    log = []
    no_batches = size // batchsize + 1
    rand_indices = torch.randperm(size, device=device)
    for i in range(no_batches):
        batch_start = i * batchsize
        batch_end = min((i + 1) * batchsize,size)
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


class NeuralNetwork(torch.nn.Module):
    def __init__(self, in_feats, no_nodes):
        super(NeuralNetwork, self).__init__()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(in_feats, no_nodes),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            # torch.nn.Linear(no_nodes,  no_nodes),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            # torch.nn.Linear(no_nodes,  no_nodes),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            # torch.nn.Linear(no_nodes,  no_nodes),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            # torch.nn.Tanh(),
            torch.nn.Linear(no_nodes, 1),
            # torch.nn.ReLU()
            # torch.nn.Linear(in_feats, 1),
            # torch.nn.ReLU()
        )

    def forward(self, x):
        return self.stack(x)


def train_ensemble(ensemble, training, test, epochs, verbose: bool = False, show_loss: bool = False):
    training_feats, training_labels = training
    test_feats, test_labels = test

    logs = []
    for modelidx, model in enumerate(ensemble):

        if verbose:
            print(f"Training model #{modelidx + 1}")
        loss_fn = torch.nn.MSELoss()
        # optimiser = torch.optim.SGD(model.parameters(), lr=1e-4)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

        log = []
        for t in range(epochs):
            curr_log = train_loop(training_feats, training_labels, model, loss_fn, optimiser,
                                  batchsize=256, verbose=False, device=device)
            log = log + curr_log

        logs.append(log)
    if show_loss:
        for log in logs:
            offset = 10 if len(log) > 10 else 0
            plt.plot(log[offset:])
        plt.title("Loss over iterations")
        plt.show()

    with torch.no_grad():
        test_labels_pred, _ = ensemble(test_feats)
        training_labels_pred, _ = ensemble(training_feats)

        score = torch.nn.MSELoss()
        train_score = torch.sqrt(score(training_labels_pred, training_labels))
        test_score = torch.sqrt(score(test_labels_pred, test_labels))
        return train_score, test_score


def cross_validation(cv_training: Tuple[torch.Tensor, torch.Tensor],
                     cv_test: Tuple[torch.Tensor, torch.Tensor],
                     epochs: int,
                     layers: Sequence[int],
                     stats: bool = True,
                     verbose: bool = False,
                     show_loss: bool = False,
                     show_importances: bool = False) -> BayesianNetworkEnsemble | Tuple[BayesianNetworkEnsemble, Dict]:
    cv_training_feats, cv_training_labels = cv_training
    cv_test_feats, cv_test_labels = cv_test
    ensembles = []

    train_scores = torch.zeros([len(cv_training_feats)], dtype=torch.float)
    test_scores = torch.zeros([len(cv_training_feats)], dtype=torch.float)
    reg = None
    importances = torch.empty((len(cv_training_feats), cv_training_feats.shape[-1]))
    for cv_idx in range(len(cv_training_feats)):
        # curr_train_feats = torch.from_numpy(sc_sk.fit_transform(cv_training_feats[j])).float()
        # curr_train_labels = torch.from_numpy(cv_training_labels[j]).float()
        # curr_test_feats = torch.from_numpy(sc_sk.transform(cv_test_feats[j])).float()
        # curr_test_labels = torch.from_numpy(cv_test_labels[j]).float()
        # ds = CoolingDataset(curr_train_feats, curr_train_labels)
        # ds = CoolingDataset(cv_training_feats[j], cv_training_labels[j])

        # train_loader = DataLoader(ds, batch_size=256, shuffle=True)

        if verbose:
            print(f"----- {nodes:03}/{cv_idx + 1:02} nodes -----")
        ensemble = BayesianNetworkEnsemble(cv_training_feats.shape[-1], layers=layers, noise_variance=0.018 ** 2)
        train_score, test_score = train_ensemble(ensemble, (cv_training_feats[cv_idx], cv_training_labels[cv_idx]), (cv_test_feats[cv_idx], cv_test_labels[cv_idx]),
                                                 epochs, show_loss=show_loss, verbose=verbose)
        train_scores[cv_idx] = train_score
        test_scores[cv_idx] = test_score
        ensembles.append(ensemble)
        importances[cv_idx] = ensemble.importance()
        if verbose:
            print(f"Importances: {importances[cv_idx]}")

    avg_importances = importances.mean(dim=0)
    avg_importances /= avg_importances.max()
    if show_importances:
        plt.bar(avg_importances)
        plt.show()

    masked_train = remove_minmax(train_scores)
    masked_test = remove_minmax(test_scores)
    if stats:
        stats = {
            'training_scores': train_scores.tolist(),
            'test_scores': test_scores.tolist(),
            'average_importances': avg_importances.tolist(),
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


def plot_true_predicted(true, predicted, predicted_err: None, title: str = None, save_fig: bool = False, show: bool = True):
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train neural network on turbine film cooling database")
    parser.add_argument("-d", "--directory", type=str, required=False, help="Database directory", default="data")
    parser.add_argument("--super", action="store_true", help="To run on a supercomputer")
    parser.add_argument("--cpu", action="store_true", help="Force running on CPU")
    parser.add_argument("--loss", action="store_true", help="Show loss curves after training")
    parser.add_argument("--plot", action="store_true", help="Plot test set predictions at end")
    parser.add_argument("--prior", action="store_true", help="Show prior distribution")
    parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs to run", default=70)
    parser.add_argument("-n", "--nodes", type=int, help="Number of nodes in each hidden layer", default=100)
    parser.add_argument("--cv", type=int, help="Number of cross-validation sets", default=5)
    parser.add_argument("--hidden", type=int, help="Number of hidden layers", default=1)
    parser.add_argument("--logx", action="store_true", help="Use log(x/D) as a feature instead of x/D")
    parser.add_argument("--comment", type=str, help="Comment to save in the run log", default="")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.super:
        print("Running in supercomputer mode")

    if args.cpu:
        print("Forcing to run on CPU")
        device = "cpu"

    print(f"Using {device} device")


    db = CoolingDatabase(Path(args.directory))
    flow_params = ["Ma", "AR", "VR", "Re", "W/D", "IR normal", "IR perpendicular"]
    epochs = args.epochs
    nodes = args.nodes
    no_hidden = args.hidden
    layers = [nodes] * no_hidden + [1]
    cv_count = args.cv
    training_min = 10000
    test_min = 2000
    is_logx = args.logx

    print(f"Using variables {flow_params}\n"
          f"Running for {epochs} epochs, with network layers: {layers}\n")

    # Use normal test-train split for plotting purposes
    db.generate_dataset(training_min, test_min, flow_param_list=flow_params)
    training_feats, training_labels = db.get_dataset()
    test_feats, test_labels = db.get_dataset(test=True)
    training_files = db.get_files()
    test_files = db.get_files(test=True)

    # Use entire dataset for cross validation
    db.generate_dataset(12000, 0, flow_param_list=flow_params)
    (cv_training_feats, cv_training_labels), (cv_test_feats, cv_test_labels) = db.get_crossvalidation_sets(cv_count, padding="random")

    cv_training_feats = np.stack(cv_training_feats)
    cv_training_labels = np.stack(cv_training_labels)
    cv_test_feats = np.stack(cv_test_feats)
    cv_test_labels = np.stack(cv_test_labels)

    # Move everything to GPU if possible
    cv_training_feats = torch.from_numpy(cv_training_feats).float().to(device)
    cv_training_labels = torch.from_numpy(cv_training_labels).float().to(device)
    cv_test_feats = torch.from_numpy(cv_test_feats).float().to(device)
    cv_test_labels = torch.from_numpy(cv_test_labels).float().to(device)

    # Log x/D
    if is_logx:
        cv_training_feats[:, :, -1] = torch.log(cv_training_feats[:, :, -1])
        cv_test_feats[:, :, -1] = torch.log(cv_test_feats[:, :, -1])
        training_feats[:, -1] = np.log(training_feats[:, -1])
        test_feats[:, -1] = np.log(test_feats[:, -1])
    # Scaling bit dodgy due to padding with zeros, TODO: fix later...
    ensemble = None
    sc = CustomStandardScaler()
    cv_training_feats = sc.fit_transform(cv_training_feats, dim=1)
    cv_test_feats = sc.transform(cv_test_feats)
    # sc_sk = StandardScaler()
    print(f"Running {cv_count}-fold cross-validation")

    cv_ensembles, cv_results = cross_validation((cv_training_feats, cv_training_labels), (cv_test_feats, cv_test_labels),
                                                epochs, layers, stats=True, verbose=True, show_loss=args.loss)
    print(f"\r{nodes:03}/{len(cv_training_feats):02} nodes, "
          f"avg. training score: {cv_results['masked_training_average']:.3g} "
          f"σ = {cv_results['masked_training_stdev']:.3g}, "
          f"avg. test score: {cv_results['masked_test_average']:.3g} "
          f"σ = {cv_results['masked_test_stdev']:.3g}")
    print(f"All training scores: {cv_results['training_scores']}\nAll test scores: {cv_results['test_scores']}")
    print(f"Average relative feature importances: {cv_results['average_importances']}")

    training_feats = sc.fit_transform(torch.from_numpy(training_feats).float().to(device), dim=0)
    training_labels = torch.from_numpy(training_labels).float().to(device)
    test_feats = sc.transform(torch.from_numpy(test_feats).float().to(device))
    test_labels = torch.from_numpy(test_labels).float().to(device)

    ensemble = BayesianNetworkEnsemble(len(training_feats[0]), layers, 0.018 ** 2)

    if args.prior:
        with torch.no_grad():
            prior_mean, prior_std = ensemble(test_feats)
            plot_true_predicted(test_labels[:, 0], prior_mean[:, 0], predicted_err=prior_std[:, 0], title="Prior")

    print("Now training with all training data")
    train_main_score, test_main_score = train_ensemble(ensemble, (training_feats, training_labels), (test_feats, test_labels),
                                                       epochs, verbose=False, show_loss=args.loss)
    print(f"Training score: {train_main_score:.3g}, " 
          f"Test score: {test_main_score:.3g}")

    with torch.no_grad():
        post_mean, post_std = ensemble(test_feats)
        plot_true_predicted(test_labels[:, 0], post_mean[:, 0], predicted_err=post_std[:, 0], title="Posterior")

    log_dict = {
        'id': RUN_ID,
        'date': datetime.utcfromtimestamp(RUN_ID).strftime("%Y-%m-%d %H:%M:%S"),
        'training_minimum': training_min,
        'test_minimum': test_min,
        'training_count': len(training_feats),
        'test_count': len(test_feats),
        'training_files': sorted([f.name for f in training_files]),
        'test_files': sorted([f.name for f in test_files]),
        'input_parameters': flow_params + ['x_D'],
        'is_logx': is_logx,
        'no_nodes': nodes,
        'no_hidden': no_hidden,
        'epochs': epochs,
        'layers': layers,
        'comment': args.comment,
        'cv_count': cv_count,
        'cv_results': cv_results,
        'all_training_result': train_main_score.item(),
        'all_test_result': test_main_score.item()
    }

    log_path = Path.cwd() / "logs" / f"run_{RUN_ID}.json"
    with open(log_path, "w") as log_file:
        json.dump(log_dict, log_file, indent=2)

    score = torch.nn.MSELoss()
    if args.plot:
        for file in test_files:
            f = Figure(file)
            feats, labels = f.get_feature_label_maps(flow_param_list=flow_params)
            study_name = file.name.split('_')[0]
            is_study_in_training = any(x.name.startswith(study_name) for x in training_files)

            fig, axes = plt.subplots(1, len(feats), figsize=(16, 9), sharey=True)

            fig.set_tight_layout(True)
            scores = []
            for feat, label, ax in zip(feats, labels, np.atleast_1d(axes)):

                # feat_scaled = torch.from_numpy(sc_sk.transform(feat)).float()

                # Resample if less than 10 datapoints
                resample_forplot = len(feat) < 10
                feat_torched = torch.from_numpy(feat).float()
                feat_to_plot = feat_torched.detach().clone()
                if resample_forplot:
                    min_x = torch.min(feat_torched[:, -1]).item()
                    max_x = torch.max(feat_torched[:, -1]).item()
                    no_dims = feat_torched.size(dim=1)
                    feat_to_plot = torch.tile(feat_torched[0], (10, 1))
                    feat_to_plot[:, -1] = torch.linspace(min_x, max_x, 10)

                if is_logx:
                    feat_to_plot[:, -1] = torch.log(feat_to_plot[:, -1])
                    feat_torched[:, -1] = torch.log(feat_torched[:, -1])
                feat_to_plot_scaled = (feat_to_plot - sc.mean[-1]) / sc.std[-1]
                feat_scaled = (feat_torched - sc.mean[-1]) / sc.std[-1]

                label_pred_mean, label_pred_std = ensemble(feat_scaled)

                label_toplot_mean, label_toplot_std = label_pred_mean.detach().clone(), label_pred_std.detach().clone()
                if resample_forplot:
                    label_toplot_mean, label_toplot_std = ensemble(feat_to_plot_scaled)
                upper, lower = label_toplot_mean + 2 * label_toplot_std, label_toplot_mean - 2 * label_toplot_std

                if is_logx:
                    feat_to_plot[:, -1] = torch.exp(feat_to_plot[:, -1])
                ax.errorbar(feat[:, -1], label, yerr=f.get_eff_uncertainty(), fmt="o", label="True value", markersize=3)
                ax.plot(feat_to_plot[:, -1], label_toplot_mean[:, 0], color="orange", label="NN predicted")
                ax.fill_between(feat_to_plot[:, -1], upper[:, 0], lower[:, 0], alpha=0.4, label="95% confidence")

                curr_score = torch.sqrt(score(torch.squeeze(label_pred_mean), torch.from_numpy(label).float()))
                scores.append(curr_score)

                ax.set_title(f"$\\bar{{\\Delta}}$: {curr_score.item():.3g}")
                ax.set_ylim((0.0, 1.0))
                # ax.legend()

            fig.suptitle(f"{f}\n"
                         f"Parameters used: {flow_params}, NN layers: {ensemble.layers}, "
                         f"$\log{{x/D}}$ as feature? {'Yes' if is_logx else 'No'}\n"
                         f"{args.comment}\n"
                         f"Is same study in training set? {'Yes' if is_study_in_training else 'No'}\n"
                         f"Average MSE: {sum(scores)/len(scores):.3g}")


            plot_ctr = 1
            plot_path = Path.cwd() / "plots" / (file.stem + f"_{RUN_ID}" + ".png")
            while plot_path.exists():
                plot_ctr += 1
                plot_path = Path.cwd() / "plots" / (file.stem + f"_{RUN_ID}" + ".png")

            fig.savefig(plot_path)
            plt.show()
