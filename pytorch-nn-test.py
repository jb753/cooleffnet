import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader

from fcdb import CoolingDatabase, Figure

def remove_minmax(array):
    mask = np.logical_or(array == array.max(keepdims=True), array == array.min(keepdims= True))
    a_masked = np.ma.masked_array(array, mask=mask)
    return a_masked

class CoolingDataset(Dataset):
    def __init__(self, feats, labels, transform=None, target_transform=None, device="cpu"):
        self.feats = torch.from_numpy(feats).to(device)
        self.labels = torch.from_numpy(labels).to(device)
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


def train_loop(dataloader, model, loss_fn, opt, verbose=False):
    size = len(dataloader.dataset)
    log = []
    for batch, (X, y) in enumerate(dataloader):
        # Predict and calc. loss
        pred = model(X.float())
        loss = loss_fn(pred, y.float())

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            if verbose:
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            log.append(loss)

    return log


class NeuralNetwork(torch.nn.Module):
    def __init__(self, in_feats, no_nodes):
        super(NeuralNetwork, self).__init__()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(in_feats, no_nodes),
            # torch.nn.ReLU(),
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

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    db = CoolingDatabase(Path(args.directory))
    flow_params = ["Ma", "AR", "VR", "Re", "W/D", "BR"]
    db.generate_dataset(10000, 2000, flow_param_list=flow_params)
    training_feats, training_labels = db.get_dataset()
    test_feats, test_labels = db.get_dataset(test=True)
    training_files = db.get_files()
    test_files = db.get_files(test=True)

    # split_feats, split_labels = db.split_training(5)
    (cv_training_feats, cv_training_labels), (cv_test_feats, cv_test_labels) = db.get_crossvalidation_sets(10)

    model = None
    sc = StandardScaler()

    for i in range(30, 31, 1):

        train_scores = []
        test_scores = []
        reg = None
        for j in range(len(cv_training_feats)):
            training_feats_scaled = sc.fit_transform(cv_training_feats[j])
            test_feats_scaled = sc.transform(cv_test_feats[j])

            train_loader = DataLoader(CoolingDataset(training_feats_scaled, cv_training_labels[j]), batch_size=200, shuffle=True)


            repeat = 5
            model = NeuralNetwork(len(flow_params) + 1, i).to(device)
            loss_fn = torch.nn.MSELoss()
            #optimiser = torch.optim.SGD(model.parameters(), lr=1e-2)
            optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

            epochs = 150
            log = []
            for t in range(epochs):
                #print(f"Epoch {t + 1}")
                curr_log = train_loop(train_loader, model, loss_fn, optimiser, verbose=False)
                log = log + curr_log

            # plt.plot(log)
            # plt.show()

            # for _ in range(repeat):
            #     reg = MLPRegressor(hidden_layer_sizes=(i), activation="relu", solver="adam", learning_rate="invscaling",
            #                                             max_iter=2000, verbose=False, shuffle=True).fit(training_feats_scaled, cv_training_labels[j].ravel())
            #
            #     test_labels_pred = reg.predict(test_feats_scaled)
            #     training_labels_pred = reg.predict(training_feats_scaled)
            #
            #
            #     training_r2 += r2_score(cv_training_labels[j].ravel(),training_labels_pred.ravel())
            #     test_r2 += r2_score(cv_test_labels[j].ravel(), test_labels_pred.ravel())
            # training_r2 /= repeat
            # test_r2 /= repeat

            with torch.no_grad():
                test_labels_pred = model(torch.from_numpy(test_feats_scaled).float())
                training_labels_pred = model(torch.from_numpy(training_feats_scaled).float())

                training_r2 = mean_squared_error(cv_training_labels[j].ravel(),training_labels_pred.ravel())
                test_r2 = mean_squared_error(cv_test_labels[j].ravel(), test_labels_pred.ravel())


                train_scores.append(np.sqrt(training_r2))
                test_scores.append(np.sqrt(test_r2))

                masked_train = remove_minmax(np.asarray(train_scores))
                masked_test = remove_minmax(np.asarray(test_scores))
                train_r2_mean = np.mean(train_scores) if len(train_scores) < 3 else masked_train.mean()
                train_r2_std = np.std(train_scores) if len(train_scores) < 3 else masked_train.std()
                test_r2_mean = np.mean(test_scores) if len(test_scores) < 3 else masked_test.mean()
                test_r2_std = np.std(test_scores) if len(test_scores) < 3 else masked_test.std()


                print(f"\r{i:02}/{j+1:02} nodes, avg. training score: {train_r2_mean:.3g}, σ = {train_r2_std:.3g},"
                      f" avg. test score: {test_r2_mean:.3g}, σ = {test_r2_std:.3g}", end='')

        print("")

    #exit(0)

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
        for file in test_files:
            f = Figure(file)
            feats, labels = f.get_feature_label_maps(flow_param_list=flow_params)
            study_name = file.name.split('_')[0]
            is_study_in_training = any(x.name.startswith(study_name) for x in training_files)

            fig, axes = plt.subplots(1, len(feats), figsize=(16, 9), sharey=True)
            fig.suptitle(f"{file.name}\nIs same study in training set?{'Yes' if is_study_in_training else 'No'}")

            for feat, label, ax in zip(feats, labels, np.atleast_1d(axes)):
                feat_scaled = sc.transform(feat)

                label_pred = model(torch.from_numpy(feat_scaled).float())
                ax.errorbar(feat_scaled[:, -1], label, yerr=f.get_eff_uncertainty(), fmt="o", label="True value", markersize=3)
                ax.plot(feat_scaled[:, -1], label_pred, color="orange", label="NN predicted")
                ax.legend()

            plt.show()
