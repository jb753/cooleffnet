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
    db.generate_dataset(4000, 2000, flow_param_list=flow_params)
    training_feats, training_labels = db.get_dataset()
    test_feats, test_labels = db.get_dataset(test=True)
    training_files = db.get_files()
    test_files = db.get_files(test=True)

    sc = StandardScaler()
    training_feats_scaled = sc.fit_transform(training_feats)
    test_feats_scaled = sc.transform(test_feats)

    reg = MLPRegressor(hidden_layer_sizes=(50, 100), activation="relu",
                                            max_iter=2000, verbose=True, tol=1e-10).fit(training_feats_scaled, training_labels.ravel())

    labels_pred = reg.predict(test_feats_scaled)
    print(f"The score: {r2_score(test_labels.ravel(), labels_pred.ravel())}")

    plt.scatter(test_labels, labels_pred)
    plt.plot(test_labels, test_labels, color='red')
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
            ax.set_ylim((0.0, 0.5))

        plt.show()
