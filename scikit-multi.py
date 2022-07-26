import random
from pathlib import Path
import itertools

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots
import sklearn.gaussian_process.kernels as k
from sklearn.gaussian_process import GaussianProcessRegressor

from fcdb import Figure, CoolingDatabase

if __name__ == "__main__":
    db = CoolingDatabase(Path("data"), verbose=True)
    db.generate_dataset(1000, 100, shuffle=True)
    training_feats, training_labels = db.get_dataset()
    training_files = db.get_files(test=False)
    test_files = db.get_files(test=True)

    # fig, axes = plt.subplots(2, 3, figsize=(16,8))
    # for i, ax in enumerate(itertools.chain.from_iterable(axes)):
    #     ax.scatter(feat_list[:, i], label_list, marker='x', s=1)
    #     ax.set_title(Figure.feature_names()[i])
    #
    # plt.show()

    # fig3D = plotly.subplots.make_subplots(rows=2, cols=3,
    #                                       specs=[[{'type': "scatter3d"}, {'type': "scatter3d"}, {'type': "scatter3d"}],
    #                                              [{'type': "scatter3d"}, {'type': "scatter3d"}, {'type': "scatter3d"}]])
    #
    # for i in range(5):
    #     fig3D.add_trace(go.Scatter3d(x=feat_list[:, i], y=feat_list[:, -1], z=label_list,
    #                     # labels={'x': Figure.feature_names()[i], 'y': Figure.feature_names()[-1], 'z': "Film effectiveness"},
    #                     name=Figure.feature_names()[i],
    #                     marker=go.scatter3d.Marker(size=1),
    #                     mode='markers',
    #                     opacity=0.8),
    #                     row=i // 3 + 1,
    #                     col=i % 3 + 1)
    #
    # fig3D.show()
    # fig = plt.figure(figsize=(16, 8))
    # axes = [
    #     fig.add_subplot(2, 3, 1, projection='3d'),
    #     fig.add_subplot(2, 3, 2, projection='3d'),
    #     fig.add_subplot(2, 3, 3, projection='3d'),
    #     fig.add_subplot(2, 3, 4, projection='3d'),
    #     fig.add_subplot(2, 3, 5, projection='3d'),
    # ]
    # # ax.scatter(feat_list[:, 0], feat_list[:, -1], label_list, s=1)
    # for i, ax in enumerate(axes):
    #     ax.scatter(feat_list[:, i], feat_list[:, -1],label_list, marker='x', s=1)
    #     ax.set_title(Figure.feature_names()[i])
    #
    # plt.show()

    kernel = k.Matern(length_scale=np.ones(len(training_feats[0]))) + k.ConstantKernel() + k.WhiteKernel()
    # TODO: Change alpha to be a proper array...
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)

    gp.fit(training_feats, training_labels)

    print(f"Initial: {kernel}\nOptimum: {gp.kernel_}\nLog-Marginal-Likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta)}")

    # y_mean, y_stdev = gp.predict(test_feats, return_std=True)
    # fig, axes = plt.subplots(2, 3, figsize=(16,8))
    # for i, ax in enumerate(itertools.chain.from_iterable(axes)):
    #     ax.scatter(test_feats[:, i], test_labels, marker='x', s=1, label="True value")
    #     ax.scatter(test_feats[:, i], y_mean, marker='o', s=1, label = "Predicted")
    #     ax.legend()
    #     ax.set_title(Figure.feature_names()[i])
    # fig.suptitle(f"Initial: {kernel}\nOptimum: {gp.kernel_}\nLog-Marginal-Likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta)}")

    plt.show()
    # test_files = ["data/McNamara2021_Fig12_CO2.json", "data/Anderson2015_Fig5a.json"]

    for file in test_files:
        test_figure = Figure(file)
        feats, labels = test_figure.get_feature_label_maps()

        study_name = file.name.split('_')[0]
        is_study_in_training = any(f.name.startswith(study_name) for f in training_files)
        fig, axes = plt.subplots(1, len(feats), figsize=(16,8), sharey=True)
        fig.suptitle(f"Dataset: {file}\nIs figure from same study in in training? {'Yes' if is_study_in_training else 'No'}")

        for feat, label, ax in zip(feats, labels, np.atleast_1d(axes)):
            y_mean, y_stdev = gp.predict(feat, return_std=True)

            upper, lower = y_mean + 2 * y_stdev, y_mean - 2 * y_stdev
            ax.plot(feat[:, -1], label, label="True value")
            ax.plot(feat[:, -1], y_mean, label="GP Predicted")
            ax.fill_between(feat[:, -1], upper, lower, alpha=0.4)
        plt.legend()
        plt.show()





