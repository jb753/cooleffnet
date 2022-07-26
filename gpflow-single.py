import random
from pathlib import Path
import itertools

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots
import sklearn.gaussian_process.kernels as k
from sklearn.gaussian_process import GaussianProcessRegressor
import gpflow
import tensorflow
from gpflow.utilities import print_summary

from fcdb import Figure

if __name__ == "__main__":

    testpath = Path("data/McNamara2021_Fig12_CO2.json")
    f = Figure(testpath)

    map = f.get_feature_label_maps()
    fig, axes = plt.subplots(1, len(map[0]), figsize=(16, 8), sharey=True)
    for feats, labels, ax in zip(*map, np.atleast_1d(axes)):
        label_mean = np.mean(labels)
        set_length = len(feats)
        random_indices = np.random.choice(np.arange(set_length), set_length // 5, replace=False)
        feat_sample = feats[::20]
        label_sample = labels[::20]

        # fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        ax.errorbar(feat_sample[:, -1], label_sample, yerr=f.get_eff_uncertainty(), fmt="o", label="Sample")
        ax.plot(feats[:, -1], labels, label="Function")
        ax.set_ylim(0, 0.4)

        kernel = gpflow.kernels.RBF()
        print_summary(k)

        feat_sample_x = np.atleast_2d(feat_sample[:, -1]).T
        label_sample_y = np.atleast_2d(label_sample).T
        feat_x = np.atleast_2d(feats[:, -1]).T
        test_x = np.linspace(0, 25, 200).reshape(200, 1)

        m = gpflow.models.GPR(data=(feat_sample_x, label_sample_y), kernel=kernel, mean_function=None)
        m.likelihood.variance.assign(f.get_eff_uncertainty() ** 2)


        print_summary(m)

        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options={'maxiter':100})
        print_summary(m)

        mean, var = m.predict_f(test_x)
        upper, lower = mean[:, 0] + 2 * np.sqrt(var[:, 0]), mean[:, 0] - 2 * np.sqrt(var[:, 0])

        ax.plot(test_x[:, 0], mean[:, 0], label="GP mean")
        ax.fill_between(test_x[:, 0], upper, lower, alpha=0.4, label="95% confidence interval")
        ax.legend()

    plt.plot()
    plt.show()
