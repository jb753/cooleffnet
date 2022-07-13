from pathlib import Path

from Figure import Figure

if __name__ == "__main__":
    data_files = [path for path in Path("data").iterdir() if path.is_file() and path.suffix == ".json"]

    # TODO: Add proper testing
    data_set_no = 0
    example_no = 0
    for file in data_files:
        test = Figure(file)
        print(test.get_velocity_ratio())
        print(test.get_feature_label_maps())
        print(test.get_reynolds())
        print(test.get_mach())
        map = test.get_feature_label_maps()

        example_no += sum(len(x) for x in map)

        if type(test.get_reynolds()) is float:
            data_set_no += 1
        else:
            data_set_no += len(test.get_reynolds())

    print(f"No of examples: {example_no}")
