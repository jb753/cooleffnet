import datetime
import random
from pathlib import Path
import itertools

import gpflow
import gpflow.kernels as k
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots
import tensorflow as tf

from fcdb import Figure, CoolingDatabase
from util import run_adam

if __name__ == "__main__":

    flow_params = ["AR", "Ma", "Re", "W/D", "VR"]
    db = CoolingDatabase(Path("data"), verbose=True)
    db.generate_dataset(11000, 1000, flow_param_list=flow_params)

    training_feats, training_labels, stats = db.get_dataset(test=False, normalize_features=True, zero_mean_labels=True, return_stats=True)

    training_files = db.get_files()
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

    # Set up inducing points
    no_inducing = 100
    random_for_inducing = np.random.choice(np.arange(len(training_feats)), no_inducing, replace=False)
    inducing = training_feats[random_for_inducing, :].copy()



    # Set up starting values
    alpha = 5
    start_lengthscales = np.sqrt(np.var(training_feats, axis=0)) / alpha
    # TODO: Use something else here

    # Setup model
    kernels = gpflow.kernels.White() + gpflow.kernels.Constant()
    # for i, (lengthscale, var) in enumerate(zip(start_lengthscales, start_variances)):
        # kernels *= k.RBF(lengthscales=1.0)
    kernels += gpflow.kernels.Matern52(lengthscales=[1.0, 1.0, 1.0,1.0,1.0, 1.0]) * gpflow.kernels.Constant()

    gpflow.utilities.print_summary(kernels)

    #m = gpflow.models.GPR(data=(training_feats, training_labels), kernel=kernels, mean_function=None)
    m = gpflow.models.SVGP(kernels, gpflow.likelihoods.Gaussian(), inducing, num_data=len(training_feats), whiten=False)
    gpflow.utilities.print_summary(m)

    train_dataset = tf.data.Dataset.from_tensor_slices((training_feats, training_labels)).repeat().shuffle(len(training_feats))
    gpflow.set_trainable(m.inducing_variable, False)


    #m.likelihood.variance.assign(likelihood_var)

    # Setup monitoring
    log_dir = f"logs/run{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_task = gpflow.monitor.ModelToTensorBoard(log_dir, m)
    lml_task = gpflow.monitor.ScalarToTensorBoard(log_dir, lambda: m.training_loss(), "training_objective")
    fast_tasks = gpflow.monitor.MonitorTaskGroup([model_task, lml_task], period=1)
    monitor = gpflow.monitor.Monitor(fast_tasks)

    logf = run_adam(m, train_dataset, minibatch_size=100, iterations=20000)

    plt.plot(logf)
    plt.title("ELBO / every 100 iterations")
    plt.show()
    # opt = gpflow.optimizers.Scipy()
    # opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100), step_callback=monitor)

    gpflow.utilities.print_summary(m)



    #print(f"Initial: {kernel}\nOptimum: {gp.kernel_}\nLog-Marginal-Likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta)}")

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
        feats, labels = test_figure.get_feature_label_maps(flow_param_list=flow_params)

        study_name = file.name.split('_')[0]
        is_study_in_training = any(f.name.startswith(study_name) for f in training_files)
        fig, axes = plt.subplots(1, len(feats), figsize=(16,8), sharey=True)
        fig.suptitle(f"Dataset: {file}\nIs figure from same study in in training? {'Yes' if is_study_in_training else 'No'}")

        for feat, label, ax in zip(feats, labels, np.atleast_1d(axes)):
            feat_norm = (feat - stats['feat_means']) / stats['feat_stdevs']
            mean_tf, var_tf = m.predict_f(feat_norm)
            mean, var = mean_tf.numpy(), var_tf.numpy()
            mean[:, 0] = mean[:, 0] + stats['label_means']

            upper, lower = mean[:, 0] + 1.96 * np.sqrt(var[:, 0]), mean[:, 0] - 1.96 * np.sqrt(var[:, 0])

            ax.errorbar(feat[:, -1], label, yerr=test_figure.get_eff_uncertainty(), label="True value", fmt="o", markersize=2)
            ax.plot(feat[:, -1], mean[:, 0], label="GP Predicted")
            ax.fill_between(feat[:, -1], upper, lower, alpha=0.4)
        plt.legend()
        plt.show()