import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import gpflow.kernels as k
import gpflow
from gpflow.utilities import print_summary
import tensorflow as tf

from fcdb import Figure, CoolingDatabase
from util import run_adam


if __name__ == "__main__":
    flow_params = ["Ma"]
    db = CoolingDatabase(Path("data"), verbose=True)
    db.generate_dataset(4000, 100, unique_params=True, flow_param_list=flow_params)
    training_feats, training_labels, stats = db.get_dataset(zero_mean_labels=True, return_stats=True)

    # Plot values
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x=feat_list[:, -1], y=feat_list[:, wanted_dim], c=label_list, s=4, vmax=0.5, vmin=0.0, cmap=cm.coolwarm_r)
    # ax.set_ylabel(f"{Figure.feature_names()[wanted_dim]}")
    # ax.set_xlabel(f"{Figure.feature_names()[-1]}")
    # fig.colorbar(scatter)
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={"projection": "3d"})
    # surf = ax.scatter(feat_list[:, -1], feat_list[:, wanted_dim], label_list, c=label_list, cmap=cm.coolwarm_r, s=0.8, vmax=0.5, vmin=0.0)
    # ax.set_xlabel(f"{Figure.feature_names()[-1]}")
    # ax.set_ylabel(f"{Figure.feature_names()[wanted_dim]}")
    # ax.set_zlabel(f"$\epsilon$")
    # fig.colorbar(surf)
    # plt.show()
    #
    # tc = plt.tricontourf(feat_list[:, -1], feat_list[:, wanted_dim], label_list, cmap=cm.coolwarm_r)
    # plt.xlabel(f"{Figure.feature_names()[-1]}")
    # plt.ylabel(f"{Figure.feature_names()[wanted_dim]}")
    # plt.colorbar(tc)
    # plt.show()

    # alpha = 1
    # # start_lengthscales = np.sqrt(np.var(training_feats[:, [wanted_dim, 5]], axis=0)) / alpha
    # start_lengthscales = [1.0, 1.0]
    #
    # kernel = k.SquaredExponential(lengthscales=[start_lengthscales]) + k.Constant()
    # m = gpflow.models.GPR(data=(training_feats[:, [wanted_dim, 5]], np.atleast_2d(training_labels).T), kernel=kernel)
    # # m.likelihood.variance.assign(0.03)
    # gpflow.utilities.print_summary(m)
    #
    #
    # opt = gpflow.optimizers.Scipy()
    # opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options={'maxiter': 100})
    #
    # gpflow.utilities.print_summary(m)

    alpha = 10
    start_lengthscales = np.sqrt(np.var(training_feats, axis=0)) / alpha
    kernel = k.RBF(lengthscales=[1.0, 1.0])
    # kernel = k.SquaredExponential(lengthscales=start_lengthscales[0], active_dims=[0])\
    #          + k.SquaredExponential(lengthscales=start_lengthscales[1],active_dims=[1])

    no_inducing_points = 100
    minibatch_size = 100
    random_indices = np.random.choice(np.arange(len(training_feats)), no_inducing_points, replace=False)
    Z = training_feats[random_indices, :].copy()

    m_svgp = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=len(training_feats), whiten=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((training_feats, training_labels)).repeat().shuffle(len(training_feats))
    train_iter = iter(train_dataset.batch(minibatch_size))

    gpflow.set_trainable(m_svgp.inducing_variable, False)
    max_iter = 20000
    logf = run_adam(m_svgp, train_dataset, minibatch_size, max_iter)

    plt.plot(logf)
    plt.title("ELBO / every 100 iterations")
    plt.show()


    # Create test points
    x_range = np.linspace(0, 80, 100)
    y_range = np.linspace(0, 1.0, 100)
    x2D, y2D = np.meshgrid(y_range, x_range)
    test_feats = np.column_stack((y2D.ravel(), x2D.ravel()))

    results, var = m_svgp.predict_f(test_feats)
    results += stats['label_means']

    upper, lower = results + np.sqrt(var) * 1.96, results - np.sqrt(var) * 1.96

    x_plot, y_plot = np.reshape(test_feats[:, 0], (100, 100)), np.reshape(test_feats[:, 1], (100, 100))
    results_plot = np.reshape(results, (100, 100))
    upper_plot = np.reshape(upper, (100, 100))
    lower_plot = np.reshape(lower, (100, 100))
    # results, var = m_svgp.predict_f(training_feats)
    # results += train_mean


    print_summary(m_svgp)

    fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x_plot, y_plot, results_plot, cmap=cm.coolwarm_r)
    ax.plot_surface(x_plot, y_plot, upper_plot, color='k', alpha=0.3)
    ax.plot_surface(x_plot, y_plot, lower_plot, color='k', alpha=0.3)
    # surf = ax.scatter(test_feats[:, 0], test_feats[:, 1], results, c=results, cmap=cm.coolwarm_r, s=0.8, vmax=0.5, vmin=0.0)
    ax.scatter(training_feats[:, 1], training_feats[:, 0], training_labels + stats['label_means'], c=training_labels + stats['label_means'], cmap=cm.coolwarm_r, s=0.8, vmax=0.5, vmin=0.0)
    ax.set_xlabel(f"{Figure.feature_names(flow_params)[-1]}")
    ax.set_ylabel(f"{Figure.feature_names(flow_params)[0]}")
    ax.set_zlabel(f"$\epsilon$")
    ax.set_zlim((0, 0.5))
    fig.colorbar(surf)
    plt.show()












