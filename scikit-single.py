import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sklearn.gaussian_process.kernels
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as k
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic

from fcdb import Figure


def fun(x_vals):
    return np.exp((x_vals + 10.5) ** 0.1) + np.sin(x_vals) / (x_vals + 1) + np.cos(2.5 * x_vals ** 0.5) ** 2


if __name__ == "__main__":
    testpath = Path("data/Saumweber2012_Fig10a.json")
    f = Figure(testpath)

    map = f.get_feature_label_maps()
    fig, axes = plt.subplots(1, len(map[0]), figsize=(16,8), sharey=True)
    for feats, labels, ax in zip(*map, np.atleast_1d(axes)):

        label_mean = np.mean(labels)
        set_length = len(feats)
        random_indices = np.random.choice(np.arange(set_length), set_length // 5, replace=False)
        feat_sample = feats[::5]
        label_sample = labels[::5]

        # fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        ax.errorbar(feat_sample[:, -1], label_sample, yerr=f.get_eff_uncertainty(), fmt="o", label="Sample")
        ax.plot(feats[:, -1], labels, label="Function")
        ax.set_ylim(0, 0.4)

        kernel = k.RBF() + k.WhiteKernel()
        gp = GaussianProcessRegressor(kernel=kernel, alpha=(f.get_eff_uncertainty() ** 2), n_restarts_optimizer=10, normalize_y=False)
        gp.fit(feat_sample, label_sample)
        y_mean, y_stdev = gp.predict(feats, return_std=True)

        upper, lower = y_mean + 2 * y_stdev, y_mean - 2 * y_stdev

        ax.plot(feats[:, -1], y_mean, label="GP")
        ax.fill_between(feats[:, -1], upper, lower, alpha=0.4, label="95% confidence")
        ax.legend(ncol=1)

        ax.set_title(f"Initial: {kernel}\nOptimum: {gp.kernel_}\nLML: {gp.log_marginal_likelihood(gp.kernel_.theta):.4}",
                     fontsize=9,
                     loc="left",
                     wrap=True)


    fig.set_tight_layout(True)
    plt.tight_layout()
    plt.show()