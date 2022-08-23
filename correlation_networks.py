from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from util import CustomStandardScaler
from pytorch_nn_test import cross_validation
import fcdb

if __name__ == "__main__":
    db = fcdb.CoolingDatabase(Path("data"))
    flow_params_cylinder = ["alpha", "P_D", "BR", "DR", "Tu", "Re"]
    flow_params_shaped = ["AR", "BR", "P_D", "W_P"]
    is_logx = True
    cv_count = 10
    epochs = 150
    layers = [100, 1]
    loss = True
    nodes = 100
    db.generate_dataset(12500, 0, flow_param_list=flow_params_cylinder, data_filter="cylindrical")
    (cv_training_feats_cyl, cv_training_labels_cyl), (cv_test_feats_cyl, cv_test_labels_cyl) = db.get_crossvalidation_sets(cv_count, padding="random")
    db.generate_dataset(12500, 0, flow_param_list=flow_params_shaped, data_filter="shaped")
    (cv_training_feats_shp, cv_training_labels_shp), (cv_test_feats_shp, cv_test_labels_shp) = db.get_crossvalidation_sets(cv_count, padding="random")

    device = "cpu"
    cv_training_feats_cyl = np.stack(cv_training_feats_cyl)
    cv_training_labels_cyl = np.stack(cv_training_labels_cyl)
    cv_test_feats_cyl = np.stack(cv_test_feats_cyl)
    cv_test_labels_cyl = np.stack(cv_test_labels_cyl)

    cv_training_feats_shp = np.stack(cv_training_feats_shp)
    cv_training_labels_shp = np.stack(cv_training_labels_shp)
    cv_test_feats_shp = np.stack(cv_test_feats_shp)
    cv_test_labels_shp = np.stack(cv_test_labels_shp)

    # Move everything to GPU if possible
    cv_training_feats_cyl = torch.from_numpy(cv_training_feats_cyl).float().to(device)
    cv_training_labels_cyl = torch.from_numpy(cv_training_labels_cyl).float().to(device)
    cv_test_feats_cyl = torch.from_numpy(cv_test_feats_cyl).float().to(device)
    cv_test_labels_cyl = torch.from_numpy(cv_test_labels_cyl).float().to(device)

    cv_training_feats_shp = torch.from_numpy(cv_training_feats_shp).float().to(device)
    cv_training_labels_shp = torch.from_numpy(cv_training_labels_shp).float().to(device)
    cv_test_feats_shp = torch.from_numpy(cv_test_feats_shp).float().to(device)
    cv_test_labels_shp = torch.from_numpy(cv_test_labels_shp).float().to(device)

    # Log x/D
    if is_logx:
        cv_training_feats_cyl[:, :, -1] = torch.log(cv_training_feats_cyl[:, :, -1])
        cv_test_feats_cyl[:, :, -1] = torch.log(cv_test_feats_cyl[:, :, -1])
        cv_training_feats_shp[:, :, -1] = torch.log(cv_training_feats_shp[:, :, -1])
        cv_test_feats_shp[:, :, -1] = torch.log(cv_test_feats_shp[:, :, -1])
    ensemble = None
    sc_cyl = CustomStandardScaler()
    cv_training_feats_cyl = sc_cyl.fit_transform(cv_training_feats_cyl, dim=1)
    cv_test_feats_cyl = sc_cyl.transform(cv_test_feats_cyl)

    sc_shp = CustomStandardScaler()
    cv_training_feats_shp = sc_shp.fit_transform(cv_training_feats_shp, dim=1)
    cv_test_feats_shp = sc_shp.transform(cv_test_feats_shp)
    # sc_sk = StandardScaler()
    print(f"Running {cv_count}-fold cross-validation")


    print("*** Cylindrical holes ***")
    cv_ensembles_cyl, cv_results_cyl = cross_validation((cv_training_feats_cyl, cv_training_labels_cyl), (cv_test_feats_cyl, cv_test_labels_cyl),
                                                        epochs, layers, stats=True, verbose=True, show_loss=loss)
    print("*** Shaped holes ***")
    cv_ensembles_shp, cv_results_shp = cross_validation((cv_training_feats_cyl, cv_training_labels_cyl), (cv_test_feats_cyl, cv_test_labels_cyl),
                                                        epochs, layers, stats=True, verbose=True, show_loss=loss)
    print("*** Cylindrical holes ***")
    print(f"\r{nodes:03}/{len(cv_training_feats_cyl):02} nodes, "
          f"avg. training score: {cv_results_cyl['masked_training_average']:.3g} "
          f"σ = {cv_results_cyl['masked_training_stdev']:.3g}, "
          f"avg. test score: {cv_results_cyl['masked_test_average']:.3g} "
          f"σ = {cv_results_cyl['masked_test_stdev']:.3g}")
    print(f"All training scores: {cv_results_cyl['training_scores']}\nAll test scores: {cv_results_cyl['test_scores']}")
    print(f"Average relative feature importances: {cv_results_cyl['average_importances']}")

    print("*** Shaped holes ***")
    print("*** Shaped holes ***")
    print(f"\r{nodes:03}/{len(cv_training_feats_shp):02} nodes, "
          f"avg. training score: {cv_results_shp['masked_training_average']:.3g} "
          f"σ = {cv_results_shp['masked_training_stdev']:.3g}, "
          f"avg. test score: {cv_results_shp['masked_test_average']:.3g} "
          f"σ = {cv_results_shp['masked_test_stdev']:.3g}")
    print(f"All training scores: {cv_results_shp['training_scores']}\nAll test scores: {cv_results_shp['test_scores']}")
    print(f"Average relative feature importances: {cv_results_shp['average_importances']}")

    cyl_training_no = cv_training_feats_cyl.shape[1]
    cyl_test_no = cv_test_feats_cyl.shape[1]
    shp_training_no = cv_training_feats_cyl.shape[1]
    shp_test_no = cv_test_feats_cyl.shape[1]

    overall_training_score = (cv_results_cyl['masked_training_average'] ** 2 * cyl_training_no + cv_results_shp['masked_training_average'] ** 2 * shp_training_no)
    overall_training_score /= (cyl_training_no + shp_training_no)
    overall_training_score = np.sqrt(overall_training_score)

    overall_test_score = (cv_results_cyl['masked_test_average'] ** 2 * cyl_test_no + cv_results_shp['masked_test_average'] ** 2 * shp_test_no)
    overall_test_score /= (cyl_test_no + shp_test_no)
    overall_test_score = np.sqrt(overall_test_score)

    print(f"Overall scores: training: {overall_training_score:.3g}, test: {overall_test_score:.3g}")

