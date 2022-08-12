import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader

from fcdb import CoolingDatabase, Figure
from util import CustomMinMaxScaler, CustomStandardScaler
from bayesian_ensemble_network import BayesianNetworkEnsemble


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train neural network on turbine film cooling database")
    parser.add_argument('-d', '--directory', type=str, required=False, help='Database directory', default="data")
    parser.add_argument("--super", action="store_true", help="To run on a supercomputer")
    parser.add_argument("--cpu", action="store_true", help="Force running on CPU")


    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.super:
        print("Running in supercomputer mode")

    if args.cpu:
        print("Forcing to run on CPU")
        device = "cpu"

    print(f"Using {device} device")


    db = CoolingDatabase(Path(args.directory))
    flow_params = ["Ma", "AR", "VR", "Re", "W/D", "BR", "beta"]
    db.generate_dataset(10000, 2000, flow_param_list=flow_params)
    training_feats, training_labels = db.get_dataset()
    test_feats, test_labels = db.get_dataset(test=True)
    training_files = db.get_files()
    test_files = db.get_files(test=True)

    # split_feats, split_labels = db.split_training(5, pad_to_max=True)
    (cv_training_feats, cv_training_labels), (cv_test_feats, cv_test_labels) = db.get_crossvalidation_sets(5, padding="max")

    cv_training_feats = np.stack(cv_training_feats)
    cv_training_labels = np.stack(cv_training_labels)
    cv_test_feats = np.stack(cv_test_feats)
    cv_test_labels = np.stack(cv_test_labels)

    # Move everything to GPU if possible
    cv_training_feats = torch.from_numpy(cv_training_feats).float().to(device)
    cv_training_labels = torch.from_numpy(cv_training_labels).float().to(device)
    cv_test_feats = torch.from_numpy(cv_test_feats).float().to(device)
    cv_test_labels = torch.from_numpy(cv_test_labels).float().to(device)

    # Scaling bit dodgy due to padding with zeros, TODO: fix later...
    ensemble = None
    sc = CustomStandardScaler()
    cv_training_feats = sc.fit_transform(cv_training_feats, dim=1)
    cv_test_feats = sc.transform(cv_test_feats)
    # sc_sk = StandardScaler()
    ensembles = []
    for i in range(100, 101, 50):

        train_scores = torch.zeros([len(cv_training_feats)], dtype=torch.float)
        test_scores = torch.zeros([len(cv_training_feats)], dtype=torch.float)
        reg = None
        for j in range(len(cv_training_feats)):
            # curr_train_feats = torch.from_numpy(sc_sk.fit_transform(cv_training_feats[j])).float()
            # curr_train_labels = torch.from_numpy(cv_training_labels[j]).float()
            # curr_test_feats = torch.from_numpy(sc_sk.transform(cv_test_feats[j])).float()
            # curr_test_labels = torch.from_numpy(cv_test_labels[j]).float()
            # ds = CoolingDataset(curr_train_feats, curr_train_labels)
            # ds = CoolingDataset(cv_training_feats[j], cv_training_labels[j])

            # train_loader = DataLoader(ds, batch_size=256, shuffle=True)

            ensemble = BayesianNetworkEnsemble(len(flow_params) + 1, layers=[i, 1], noise_variance=0.018 ** 2)

            logs = []
            for modelidx, model in enumerate(ensemble):

                print(f"------ Training model #{modelidx + 1} ------")
                loss_fn = torch.nn.MSELoss()
                # optimiser = torch.optim.SGD(model.parameters(), lr=1e-4)
                optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

                epochs = 70
                log = []
                for t in range(epochs):
                    # print(f"Epoch {t + 1}")
                    curr_log = train_loop(cv_training_feats[j], cv_training_labels[j], model, loss_fn, optimiser,
                                          batchsize=256, verbose=False, device=device)
                    log = log + curr_log

                logs.append(log)
            # for log in logs:
            #     plt.plot(log[10:])
            # plt.show()

            with torch.no_grad():
                test_labels_pred, _ = ensemble(cv_test_feats[j])
                training_labels_pred, _ = ensemble(cv_training_feats[j])

                score = torch.nn.MSELoss()
                train_scores[j] = torch.sqrt(score(training_labels_pred, cv_training_labels[j]))
                test_scores[j] = torch.sqrt(score(test_labels_pred, cv_test_labels[j]))

                # masked_train = remove_minmax(np.asarray(train_scores))
                # masked_test = remove_minmax(np.asarray(test_scores))
                # train_r2_mean = np.mean(train_scores) if len(train_scores) < 3 else masked_train.mean()
                # train_r2_std = np.std(train_scores) if len(train_scores) < 3 else masked_train.std()
                # test_r2_mean = np.mean(test_scores) if len(test_scores) < 3 else masked_test.mean()
                # test_r2_std = np.std(test_scores) if len(test_scores) < 3 else masked_test.std()

                print(f"\r{i:03}/{j+1:02} nodes", end="")
            ensembles.append(ensemble)

        masked_train = remove_minmax(train_scores)
        masked_test = remove_minmax(test_scores)
        test_avg = masked_train.mean().item()
        train_avg = masked_train.mean().item()

        ensemble = ensembles[find_nearest_idx(train_scores, test_avg)]
        print(f"\r{i:03}/{len(cv_training_feats):02} nodes, "
              f"avg. training score: {masked_train.mean().item():.3g} "
              f"σ = {masked_train.std().item():.3g}, "
              f"avg. test score: {masked_test.mean().item():.3g} "
              f"σ = {masked_test.std().item():.3g}")
        print(f"All training scores: {masked_train}\nAll test scores: {masked_test}")

    # exit(0)

    # fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    #
    # axes[0].set_title(f"Training\n$R^2$ = {training_r2}")
    # axes[0].scatter(training_labels, training_labels_pred)
    # axes[0].plot(training_labels, training_labels, color='red')
    #
    # axes[1].set_title(f"Test\n$R^2$ = {test_r2}")
    # axes[1].scatter(test_labels, test_labels_pred)
    # axes[1].plot(test_labels, test_labels, color='red')
    # plt.show()

    with torch.no_grad():
        score = torch.nn.MSELoss()
        for file in test_files:
            f = Figure(file)
            feats, labels = f.get_feature_label_maps(flow_param_list=flow_params)
            study_name = file.name.split('_')[0]
            is_study_in_training = any(x.name.startswith(study_name) for x in training_files)

            fig, axes = plt.subplots(1, len(feats), figsize=(16, 9), sharey=True)
            fig.suptitle(f"{f}\n"
                         f"Parameters used: {flow_params}, NN layers: {ensemble.layers}\n"
                         f"Is same study in training set? {'Yes' if is_study_in_training else 'No'}")

            fig.set_tight_layout(True)
            for feat, label, ax in zip(feats, labels, np.atleast_1d(axes)):

                # feat_scaled = torch.from_numpy(sc_sk.transform(feat)).float()

                # Resample if less than 10 datapoints
                resample_forplot = len(feat) < 10
                feat_torched = torch.from_numpy(feat).float()
                feat_to_plot = feat_torched
                if resample_forplot:
                    min_x = torch.min(feat_torched[:, -1]).item()
                    max_x = torch.max(feat_torched[:, -1]).item()
                    no_dims = feat_torched.size(dim=1)
                    feat_to_plot = torch.tile(feat_torched[0], (10, 1))
                    feat_to_plot[:, -1] = torch.linspace(min_x, max_x, 10)

                feat_to_plot_scaled = (feat_to_plot - sc.mean[-1]) / sc.std[-1]
                feat_scaled = (feat_torched - sc.mean[-1]) / sc.std[-1]


                label_pred_mean, label_pred_std = ensemble(feat_scaled)

                label_toplot_mean, label_toplot_std =  label_pred_mean, label_pred_std
                if resample_forplot:
                    label_toplot_mean, label_toplot_std = ensemble(feat_to_plot_scaled)
                upper, lower = label_toplot_mean + 2 * label_toplot_std, label_toplot_mean - 2 * label_toplot_std
                ax.errorbar(feat[:, -1], label, yerr=f.get_eff_uncertainty(), fmt="o", label="True value", markersize=3)
                ax.plot(feat_to_plot[:, -1], label_toplot_mean[:, 0], color="orange", label="NN predicted")
                ax.fill_between(feat_to_plot[:, -1], upper[:, 0], lower[:, 0], alpha=0.4, label="95% confidence")

                curr_score = torch.sqrt(score(torch.squeeze(label_pred_mean), torch.from_numpy(label).float()))

                ax.set_title(f"$\\bar{{\\Delta}}$: {curr_score.item():.3g}")
                ax.set_ylim((0.0, 1.0))
                #ax.legend()

            # FIXME: Very lazy implementation
            plot_ctr = 1
            plot_path = Path.cwd() / "plots" / (file.stem + f"_{plot_ctr:03}" + ".png")
            while plot_path.exists():
                plot_ctr += 1
                plot_path = Path.cwd() / "plots" / (file.stem + f"_{plot_ctr:03}" + ".png")

            fig.savefig(plot_path)
            plt.show()
