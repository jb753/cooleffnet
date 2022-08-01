from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import sklearn

from fcdb import CoolingDatabase, Figure

if __name__ == "__main__":

    db = CoolingDatabase(Path("new_data"))
    flow_params = ["Ma", "AR", "VR", "Re", "W/D", "BR"]
    db.generate_dataset(11000, 2000, flow_param_list=flow_params)
    training_feats, training_labels = db.get_dataset()
    test_feats, test_labels = db.get_dataset(test=True)
    training_files = db.get_files()
    test_files = db.get_files(test=True)

    # split_feats, split_labels = db.split_training(5)
    (cv_training_feats, cv_training_labels), (cv_test_feats, cv_test_labels) = db.get_crossvalidation_sets(5)

    sc = StandardScaler()

    for i in range(20, 100, 10):

        train_scores = []
        test_scores = []
        reg = None
        for j in range(len(cv_training_feats)):
            training_feats_scaled = sc.fit_transform(cv_training_feats[j])
            test_feats_scaled = sc.transform(cv_test_feats[j])

            repeat = 5
            training_r2 = 0
            test_r2 = 0

            for _ in range(repeat):
                reg = MLPRegressor(hidden_layer_sizes=(i), activation="relu", solver="adam", learning_rate="invscaling",
                                                        max_iter=2000, verbose=False, shuffle=True).fit(training_feats_scaled, cv_training_labels[j].ravel())

                test_labels_pred = reg.predict(test_feats_scaled)
                training_labels_pred = reg.predict(training_feats_scaled)


                training_r2 += r2_score(cv_training_labels[j].ravel(),training_labels_pred.ravel())
                test_r2 += r2_score(cv_test_labels[j].ravel(), test_labels_pred.ravel())

            training_r2 /= repeat
            test_r2 /= repeat


            train_scores.append(training_r2)
            test_scores.append(test_r2)

            train_r2_mean = np.mean(train_scores)
            train_r2_std = np.std(train_scores)
            test_r2_mean = np.mean(test_scores)
            test_r2_std = np.std(test_scores)


            print(f"\r{i:02}/{j+1:02} nodes, avg. training score: {train_r2_mean:.3f}, σ = {train_r2_std:.3f},"
                  f" avg. test score: {test_r2_mean:.3f}, σ = {test_r2_std:.3f}", end='')

        print("")

    exit(0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

    axes[0].set_title(f"Training\n$R^2$ = {training_r2}")
    axes[0].scatter(training_labels, training_labels_pred)
    axes[0].plot(training_labels, training_labels, color='red')

    axes[1].set_title(f"Test\n$R^2$ = {test_r2}")
    axes[1].scatter(test_labels, test_labels_pred)
    axes[1].plot(test_labels, test_labels, color='red')
    plt.show()

    for file in test_files:
        f = Figure(file)
        feats, labels = f.get_feature_label_maps(flow_param_list=flow_params)
        study_name = file.name.split('_')[0]
        is_study_in_training = any(x.name.startswith(study_name) for x in training_files)

        fig, axes = plt.subplots(1, len(feats), figsize=(16, 9), sharey=True)
        fig.suptitle(f"{file.name}\nIs same study in training set?{'Yes' if is_study_in_training else 'No'}")

        for feat, label, ax in zip(feats, labels, np.atleast_1d(axes)):
            feat_scaled = sc.transform(feat)

            label_pred = reg.predict(feat_scaled)
            ax.errorbar(feat_scaled[:, -1], label, yerr=f.get_eff_uncertainty(), fmt="o", label="True value", markersize=3)
            ax.plot(feat_scaled[:, -1], label_pred, color="orange", label="NN predicted")
            ax.legend()

        plt.show()
