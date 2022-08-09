from pathlib import Path

from fcdb import Figure

if __name__ == "__main__":
    data_files = [path for path in Path("data").iterdir() if path.is_file() and path.suffix == ".json"]

    # TODO: Add proper testing
    data_set_no = 0
    example_no = 0
    weighted_uncertainty = 0
    for file in data_files:
        test = Figure(file)
        map = test.get_feature_label_maps()

        figure_example_no = 0
        for i, (feat, label) in enumerate(zip(map[0], map[1])):
            example_no += len(feat)
            figure_example_no += len(feat)
            print(f"File: {file.name:30}/{i + 1:02}, length: {len(feat):10}")
        weighted_uncertainty += figure_example_no * test.get_eff_uncertainty()

    weighted_uncertainty /= example_no


    print(f"No of examples: {example_no}")
    print(f"Average uncertainty weighted by number of datapoints: {weighted_uncertainty:.3f}")
