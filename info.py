import math
from pathlib import Path
import numpy as np

from fcdb import Figure

if __name__ == "__main__":
    data_files = [path for path in Path("data").iterdir() if path.is_file() and path.suffix == ".json"]

    # TODO: Add proper testing
    data_set_no = 0
    example_no = 0
    weighted_uncertainty = 0
    sum_squared = 0
    squared_no = 0
    for file in data_files:
        test = Figure(file)
        map = test.get_feature_label_maps(include_correlations=True)

        figure_example_no = 0
        for i, (feat, label) in enumerate(zip(map[0], map[1])):
            example_no += len(feat)
            figure_example_no += len(feat)
            sum_squared += np.nansum((label[:, 0] - label[:, 1]) ** 2)
            squared_no += np.count_nonzero(~np.isnan(label[:, 1]))
            print(f"File: {file.name:30}/{i + 1:02}, length: {len(feat):10}")
        weighted_uncertainty += figure_example_no * test.get_eff_uncertainty()

    mse = sum_squared / squared_no
    weighted_uncertainty /= example_no


    print(f"No of examples: {example_no}")
    print(f"Average uncertainty weighted by number of datapoints: {weighted_uncertainty:.3f}")
    print(f"Number non-nan correlations: {squared_no}")
    print(f"Root mean squared error of correlations: {np.sqrt(mse)}")
