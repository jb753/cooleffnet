import itertools
import json
import math
import random
from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np
import CoolProp.CoolProp as CoolProp

import util
import correlations


def to_array(parameter, length):
    """Utility function to convert every parameter to arrays of the right length"""
    if type(parameter) is list:
        if len(parameter) == length:
            return parameter
        else:
            raise ValueError(f"Parameter in list form with length {len(parameter)}, "
                             f"which does not match desired length {length}")
    else:
        return np.full(length, parameter)


class CoolingDatabase:

    def __init__(self, database_dir: Path, verbose: bool = False):
        """
        An object to generate training and test datasets or cross-validation datasets from
        a set of JSON measurement data files.
        Generate dataset first by calling generate_dataset() on the database object.

        Parameters
        ----------
        database_dir: Path
            Path object pointing to the directory containing the JSON measurement data files
        verbose: bool, optional
            Boolean flag controlling verbosity (default False)
        """
        self.__datafiles = [f for f in database_dir.iterdir() if f.is_file() and f.suffix == ".json"]
        self.__figures = [Figure(f) for f in self.__datafiles]
        self.__verbose = verbose
        self.__dataset_generated = False

        self.__training_feats = None
        self.__training_labels = None
        self.__training_files = None
        self.__test_feats = None
        self.__test_labels = None
        self.__test_files = None

        self.__training_total = 0
        self.__test_total = 0

    def generate_dataset(self, training_min: int, test_min: int, unique_params: bool = False, shuffle: bool = True,
                         flow_param_list: Sequence[str] = None, include_correlations: bool = False,
                         data_filter: str = None, x_norm: str = None):
        """
        Generates training and test datasets,  so they can be queried with get_dataset() or get_crossvalidation_sets()

        Parameters
        ----------
        training_min : int
            Minimum number of examples in training set
        test_min : int
            Minimum number of examples in test set
        unique_params : bool, optional
            Boolean flag controlling whether only unique combinations of flow parameters should be included
            (default False)
        shuffle : bool, optional
            Boolean flag controlling whether individual curves should be shuffled (default True)
        flow_param_list : iterable of str, default: ["AR", "W/D", "Re", "Ma", "VR"]
            Parameters to be included in features
        include_correlations : bool, optional
            Boolean flag controlling whether correlation results should be included as a second column of labels
        data_filter : { "cylindrical", "shaped" }, optional
            Filters data based on hole shape
        x_norm : { "log", "reciprocal" }, optional
            Applies transformation to x/D. `log` takes the logarithm of x/D
            while `reciprocal` uses 1/(x/D) * ER/(P/D)
        """
        if flow_param_list is None:
            flow_param_list = ["AR", "W/D", "Re", "Ma", "VR"]

        if self.__verbose:
            print("Generating dataset...")
            print(f"Wanted features: {Figure.feature_names(list(map(Figure.to_param,flow_param_list)))}")

        self.__training_feats = []
        self.__training_labels = []
        self.__training_files = []
        self.__test_feats = []
        self.__test_labels = []
        self.__test_files = []

        figure_list = self.__figures.copy()
        if shuffle:
            random.shuffle(figure_list)

        unique_vals = set()
        train_count = 0
        test_count = 0
        for fig in figure_list:
            feats, labels = fig.get_feature_label_maps(flow_param_list, include_correlations, data_filter, x_norm)
            no_data_points = sum(len(x) for x in feats)

            if unique_params:
                for feat, label in zip(feats, labels):
                    curve_length = len(feat)
                    # Hope flow params don't change with x/D
                    # Which should be the case for the database
                    # -1 because we exclude x/D
                    if tuple(feat[0][:-1]) not in unique_vals:
                        unique_vals.add(tuple(feat[0]))
                        if train_count < training_min:
                            self.__training_feats.append(feat)
                            self.__training_labels.append(label)
                            train_count += curve_length
                        elif test_count < test_min:
                            self.__test_feats.append(feat)
                            self.__test_labels.append(label)
                            test_count += curve_length
            else:
                if train_count < training_min:
                    self.__training_feats += feats
                    self.__training_labels += labels
                    train_count += no_data_points
                    self.__training_files.append(fig.get_file())
                elif test_count < test_min:
                    self.__test_feats += feats
                    self.__test_labels += labels
                    self.__test_files.append(fig.get_file())
                    test_count += no_data_points

        self.__training_total = train_count
        self.__test_total = test_count

        if self.__verbose:
            print(f"Files loaded: {len(self.__training_files) + len(self.__test_files)}")
            print(f"Training examples: {train_count}, test examples: {test_count}")

            training_filenames = [f.name for f in self.__training_files]
            test_filenames = [f.name for f in self.__test_files]
            training_filenames = sorted(training_filenames)
            test_filenames = sorted(test_filenames)

            print(f"Training files                Test files")
            for train_file, test_file in itertools.zip_longest(training_filenames, test_filenames, fillvalue=""):
                print(f"{train_file:<30}{test_file:<30}")
        self.__dataset_generated = True

    def split_training(self, no_splits: int, padding: str = None) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        """
        Generates roughly (or exactly depending on padding) equal lists of datapoints from the training datasets,
        so that one curve does not get split into two
        Parameters
        ----------
        no_splits : int
            Number of splits
        padding : { "min", "max", "random" }, optional
            If not None then makes the splits equal length, by some method of truncating or padding.
            See `get_crossvalidation_sets()` for details
        Returns
        -------
        tuple
            A tuple of two lists containing 2D arrays of features and labels respectively
        """
        if not self.__dataset_generated:
            raise RuntimeError("Requested training dataset without calling generate_dataset() before")
        if no_splits < 2:
            raise ValueError("Number of splits should be at least 2, otherwise no need to split data")
        target = self.__training_total // no_splits
        feat_sets, label_sets = [], []

        # To alternate between under and overshooting the target
        # Makes for more equal splits
        overfill = True
        curr_feat_set, curr_label_set = [], []
        for feat, label in zip(self.__training_feats, self.__training_labels):
            no_datapoints = len(feat)
            curr_feat_len = sum(len(x) for x in curr_feat_set)
            if curr_feat_len < target:
                if not overfill and (curr_feat_len + no_datapoints) > target:
                    feat_sets.append(curr_feat_set)
                    label_sets.append(curr_label_set)
                    curr_feat_set = [feat]
                    curr_label_set = [label]
                    overfill = not overfill
                else:
                    curr_feat_set.append(feat)
                    curr_label_set.append(label)
            else:
                feat_sets.append(curr_feat_set)
                label_sets.append(curr_label_set)
                curr_feat_set = [feat]
                curr_label_set = [label]
                overfill = not overfill

        feat_sets.append(curr_feat_set)
        label_sets.append(curr_label_set)

        while len(feat_sets) > no_splits:
            feat_sets[-2] = feat_sets[-2] + feat_sets[-1]
            feat_sets.pop()
            label_sets[-2] = label_sets[-2] + label_sets[-1]
            label_sets.pop()

        lengths = [sum(len(y) for y in x) for x in feat_sets]
        if padding == "max":
            largest_set_size = max(lengths)
            for feat_set, label_set, length in zip(feat_sets, label_sets, lengths):
                padding_length = largest_set_size - length
                if padding_length > 0:
                    feat_set.append(np.zeros((padding_length, feat_set[0].shape[-1])))
                    label_set.append(np.zeros(padding_length))

        feat_sets = [np.concatenate(x) for x in feat_sets]
        label_sets = [np.atleast_2d(np.concatenate(x)) for x in label_sets]
        if label_sets[0].shape[0] == 1:
            label_sets = [x.T for x in label_sets]

        if padding == "min":
            min_set_size = min(lengths)
            feat_sets = [x[:min_set_size] for x in feat_sets]
            label_sets = [x[:min_set_size] for x in label_sets]

        if padding == "random":
            largest_set_size = max(lengths)
            for i, (feat_set, label_set, length) in enumerate(zip(feat_sets, label_sets, lengths)):
                padding_length = largest_set_size - length
                padding_indices = np.random.choice(length, padding_length)
                padding_feats = feat_set[padding_indices]
                padding_labels = label_set[padding_indices]

                feat_sets[i] = np.concatenate((feat_set, padding_feats))
                label_sets[i] = np.concatenate((label_set, padding_labels))

        return feat_sets, label_sets

    def get_crossvalidation_sets(self, no_splits: int = 5, padding: str = None) \
            -> Tuple[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Generates lists of matrices to be used in cross-validation.
        Parameters
        ----------
        no_splits : int, default : 5
            Number of cross-validation sets
        padding : { "min", "max", "random" }, optional
            If not None then makes the splits equal length, by some method of truncating or padding.
            `min` truncates every split that is longer than the shortest
            `max` pads every split that is smaller than the largest with all features and labels being 0.
            `random` pads every split that is smaller than the largest with repeating random already existing examples.

        Returns
        -------
        tuple
            A tuple of two tuples containing lists of 2D matrices of features and labels
            for training and test respectively
        """
        if not self.__dataset_generated:
            raise RuntimeError("Requested cross-validation dataset without calling generate_dataset() before")
        if no_splits < 2:
            raise ValueError("Number of cross-validation sets should be at least 2")
        feat_splits, label_splits = self.split_training(no_splits, padding)

        cv_train_feats, cv_train_labels = [], []
        cv_test_feats, cv_test_labels = [], []

        temp_feats, temp_labels = [], []
        # i selects CV test set
        # j iterates through all splits
        for i in range(no_splits):
            for j in range(no_splits):
                if i == j:
                    cv_test_feats.append(feat_splits[i])
                    cv_test_labels.append(label_splits[i])
                else:
                    temp_feats.append(feat_splits[j])
                    temp_labels.append(label_splits[j])

            cv_train_feats.append(np.concatenate(temp_feats))
            cv_train_labels.append(np.concatenate(temp_labels))
            temp_feats = []
            temp_labels = []

        return (cv_train_feats, cv_train_labels), (cv_test_feats, cv_test_labels)

    # TODO: Rethink this, are there options for making it stateless?
    def get_dataset(self, test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to get the generated training and test datasets.
        Parameters
        ----------
        test: bool, optional
            If True, returns the test dataset

        Returns
        -------
        tuple
            Tuple of two 2D matrices containing the features and the labels respectively

        """
        if not self.__dataset_generated:
            raise RuntimeError("Requested training dataset without calling generate_dataset() before")

        feat_matrix = np.concatenate(self.__training_feats) if not test else np.concatenate(self.__test_feats)
        label_matrix = np.concatenate(self.__training_labels) if not test else np.concatenate(self.__test_labels)
        label_matrix = np.atleast_2d(label_matrix)
        if label_matrix.shape[0] == 1:
            label_matrix = label_matrix.T

        return feat_matrix, label_matrix

    def get_holdout_files(self, test: bool = False) -> List[Path]:
        """
        Parameters
        ----------
        test: bool, default: False
            If True returns test files instead of training files.
        Returns
        -------
        list
            List of Paths pointing to the training or test files
        """
        if not self.__dataset_generated:
            raise RuntimeError("Requested training dataset without calling generate_dataset() before")
        else:
            return self.__test_files if test else self.__training_files

    def get_all_files(self) -> List[Path]:
        """
        Returns list of all files in database.
        Returns
        -------
        list
            List of Paths pointing to all files in the database
        """
        return self.__datafiles

    def get_example_count(self) -> int:
        """Returns number of examples in database"""
        return sum(fig.get_example_count() for fig in self.__figures)


class Figure:

    # Lookup table for aliases for parameters
    __ALIAS_TO_PARAM = {
        "ar": "AR",
        "area ratio": "AR",
        "w/d": "W_D",
        "w_d": "W_D",
        "coverage ratio": "W_D",
        "w/p": "W_P",
        "w_p": "W_P",
        "p/d": "P_D",
        "p_d": "P_D",
        "l/d": "L_D",
        "l_d": "L_D",
        "alpha": "Alpha",
        "inclination angle": "Alpha",
        "beta": "Beta",
        "compound angle": "Beta",
        "orientation angle": "Beta",
        "re": "Re",
        "reynolds": "Re",
        "reynolds number": "Re",
        "ma": "Ma",
        "mach": "Ma",
        "mach number": "Ma",
        "tu": "Tu",
        "tu number": "Tu",
        "turbulence intensity": "Tu",
        "vr": "VR",
        "velocity ratio": "VR",
        "br": "BR",
        "br normal": "BR_normal",
        "br_normal": "BR_normal",
        "normal blowing ratio": "BR_normal",
        "br perpendicular": "BR_perpendicular",
        "br_perpendicular": "BR_perpendicular",
        "perpendicular blowing ratio": "BR_perpendicular",
        "dr": "DR",
        "density ratio": "DR",
        "ir": "IR",
        "momentum flux ratio": "IR",
        "ir eff": "IR_eff",
        "ir_eff": "IR_eff",
        "effective momentum flux ratio": "IR_eff",
        "ir normal": "IR_normal",
        "ir_normal": "IR_normal",
        "normal momentum flux ratio": "IR_normal",
        "ir perpendicular": "IR_perpendicular",
        "ir_perpendicular": "IR_perpendicular",
        "perpendicular momentum flux ratio": "IR_perpendicular",
        "er": "ER",
        "tr": "TR",
        "single": "Single_hole",
        "single hole": "Single_hole",
        "is single hole": "Single_hole",
    }

    # Lookup table for readable versions of parameters
    __PARAM_TO_READABLE = {
        "AR": "Area ratio",
        "W_D": "Coverage ratio",
        "W_P": "W/P",
        "P_D": "Pitch to diameter ratio",
        "L_D": "L/D",
        "alpha": "Inclination angle",
        "beta": "Compound angle",
        "Re": "Reynolds number",
        "Ma": "Mach number",
        "Tu": "Turbulence intensity",
        "VR": "Velocity ratio",
        "BR": "Blowing ratio",
        "BR_normal": "Normal blowing ratio",
        "BR_perpendicular": "Perpendicular blowing ratio",
        "DR": "Density ratio",
        "IR": "Momentum flux ratio",
        "IR_eff": "Effective momentum flux ratio",
        "IR_normal": "Normal momentum flux ratio",
        "IR_perpendicular": "Perpendicular momentum flux ratio",
        "ER": "Advective capacity ratio",
        "TR": "Temperature ratio",
        "Single_hole": "Is single hole?",
    }

    def __init__(self, file: Path):
        """
        Initialises a Figure object from a JSON figure file
        Parameters
        ----------
        file: Path
            A Path object pointing to the JSON measurement data
        """

        self.__file = file
        with open(file) as figure_file:
            self.__figure_dict = json.load(figure_file)

            self.__variations_in = []
            # Convert every list to a numpy array
            for top_key in self.__figure_dict.keys():
                for key, value in self.__figure_dict[top_key].items():
                    if type(value) is list and type(value) is not str:
                        if top_key == 'distributions':
                            if type(value[0]) is not list:
                                # If not list of lists, turn into list of as single list
                                self.__figure_dict[top_key][key] = [self.__figure_dict[top_key][key]]

                            # Convert list of lists to list of NumPy arrays
                            # Now all distributions are lists of NumPy arrays
                            self.__figure_dict[top_key][key] = [np.asarray(x) for x in self.__figure_dict[top_key][key]]
                        else:
                            # If not distribution, just convert list to NumPy array
                            self.__variations_in.append(key)
                            self.__figure_dict[top_key][key] = np.asarray(self.__figure_dict[top_key][key])

            self.__is_single_hole = self.__figure_dict['geometry']['is_single_hole']
            self.__alpha = self.__figure_dict['geometry']['alpha']
            self.__beta = self.__figure_dict['geometry']['beta']
            self.__phi = self.__figure_dict['geometry']['phi']
            self.__psi = self.__figure_dict['geometry']['psi']
            self.__Lphi_D = self.__figure_dict['geometry']['Lphi_D']
            self.__Lpsi_D = self.__figure_dict['geometry']['Lpsi_D']
            self.__P_D = self.__figure_dict['geometry']['P_D']
            self.__L_D = self.__figure_dict['geometry']['L_D']
            self.__is_x_origin_trailing_edge = self.__figure_dict['geometry']['is_x_origin_trailing_edge']

            # Check if Vinf exists
            self.__Vinf = self.__figure_dict['dimensional']['Vinf'] \
                if 'Vinf' in self.__figure_dict['dimensional'] else None
            self.__Tc = self.__figure_dict['dimensional']['Tc'] \
                if 'Tc' in self.__figure_dict['dimensional'] else None
            self.__Tinf = self.__figure_dict['dimensional']['Tinf'] \
                if 'Tinf' in self.__figure_dict['dimensional'] else None
            self.__D = self.__figure_dict['dimensional']['D']

            self.__DR = self.__figure_dict['flow']['DR']
            self.__BR = self.__figure_dict['flow']['BR']
            self.__Tu = self.__figure_dict['flow']['Tu']
            self.__del_D = self.__figure_dict['flow']['del_D']
            self.__Lam_D = self.__figure_dict['flow']['Lam_D']
            self.__H = self.__figure_dict['flow']['H']
            self.__Reinf = self.__figure_dict['flow']['Reinf']
            # Check if Mainf exists
            self.__Mainf = self.__figure_dict['flow']['Mainf'] if 'Mainf' in self.__figure_dict['flow'] else None
            self.__coolant = self.__figure_dict['flow']['coolant']
            self.__mainstream = self.__figure_dict['flow']['mainstream']

            self.__doi = self.__figure_dict['metadata']['doi']
            self.__ref = self.__figure_dict['metadata']['ref']
            self.__fig = self.__figure_dict['metadata']['fig']
            self.__comment = self.__figure_dict['metadata']['comment']
            self.__uncertainty_eff_abs = self.__figure_dict['metadata']['uncertainty_eff_abs']

            self.__x_D = self.__figure_dict['distributions']['x_D']
            self.__eff = self.__figure_dict['distributions']['eff']

            # Fix coordinate system origin at center of hole
            if self.__is_x_origin_trailing_edge:
                offset = util.trailing_edge_offset(self.__alpha, self.__psi, self.__Lpsi_D)
                if type(offset) is np.ndarray:
                    self.__x_D = [curr_x + curr_offset for curr_x, curr_offset in zip(self.__x_D, offset)]
                else:
                    if type(self.__x_D) is np.ndarray:
                        self.__x_D += offset
                    elif type(self.__x_D) is list:
                        self.__x_D = [x + offset for x in self.__x_D]

            # Fill out None values:
            self.__Mc = CoolProp.PropsSI("molar_mass", self.__coolant)
            self.__Minf = CoolProp.PropsSI("molar_mass", self.__mainstream)

            self.__MR = self.__Mc / self.__Minf  # Molar mass ratio
            self.__IR = self.__BR * self.__BR / self.__DR  # Momentum flux ratio
            # Apparently universal gas constant can be different for different fluids in CoolProp
            # Not very universal anymore now, is it?
            self.__Rc = CoolProp.PropsSI("gas_constant", self.__coolant) / self.__Mc
            self.__Rinf = CoolProp.PropsSI("gas_constant", self.__mainstream) / self.__Minf

            # Assuming ideal gas
            self.__TR = self.__MR / self.__DR  # Coolant to mainstream temperature ratio
            if self.__Tc is None:
                self.__Tc = self.__Tinf * self.__TR
            if self.__Tinf is None:
                self.__Tinf = self.__Tc / self.__TR

            # Ideal gas, hence Cp is independent of pressure --> use atmospheric pressure
            self.__Cpc = CoolProp.PropsSI("Cp0mass", "T", self.__Tc, "P", 1e5, self.__coolant)
            self.__Cpinf = CoolProp.PropsSI("Cp0mass", "T", self.__Tinf, "P", 1e5, self.__mainstream)

            self.__Gammac = self.__Cpc / (self.__Cpc - self.__Rc)
            self.__Gammainf = self.__Cpinf / (self.__Cpinf - self.__Rinf)

            self.__Ac = np.sqrt(self.__Gammac * self.__Rc * self.__Tc)
            self.__Ainf = np.sqrt(self.__Gammainf * self.__Rinf * self.__Tinf)

            if self.__Mainf is None:
                self.__Mainf = self.__Vinf / self.__Ainf

            if self.__Vinf is None:
                self.__Vinf = self.__Mainf * self.__Ainf

            # Vinf now definitely set

            # Assuming viscosity is mostly a function of temperature --> use atmospheric pressure
            mu_inf_temp = CoolProp.PropsSI("viscosity", "T", self.__Tinf, "P", 1e5, self.__mainstream)
            self.__rhoinf = self.__Reinf * mu_inf_temp / (self.__Vinf * self.__D)

            # Now state fixed with rho
            # Do 4 extra iteration to be safer
            # By checking manually, this should be more than enough
            for i in range(1, 4):
                mu_inf_temp = CoolProp.PropsSI("viscosity", "T", self.__Tinf, "D", self.__rhoinf, self.__mainstream)
                self.__rhoinf = self.__Reinf * mu_inf_temp / (self.__Vinf * self.__D)
            self.__rhoc = self.__DR * self.__rhoinf

            self.__parameters = {}
            self.__initialise_parameters()

    def __initialise_parameters(self):
        """Initialises self.__parameters with all parameters that can be possibly relevant"""

        max_length = len(self.__x_D) if type(self.__x_D) is list else 1
        AR, edge_offset, W_D = util.get_geometry(self.__phi, self.__psi, self.__Lphi_D, self.__Lpsi_D, self.__alpha)
        self.__parameters['AR'] = to_array(AR, max_length)
        self.__parameters['edge_offset'] = to_array(edge_offset, max_length)
        self.__parameters['W_D'] = to_array(W_D, max_length)
        self.__parameters['P_D'] = to_array(self.__P_D, max_length)
        self.__parameters['W_P'] = to_array(W_D / self.__P_D, max_length)
        self.__parameters['L_D'] = to_array(self.__L_D, max_length)
        self.__parameters['Alpha'] = to_array(self.__alpha, max_length)
        self.__parameters['Beta'] = to_array(np.sin(np.radians(self.__beta)), max_length)
        self.__parameters['Re'] = to_array(self.get_reynolds(), max_length)
        self.__parameters['Ma'] = to_array(self.get_mach(), max_length)
        self.__parameters['Tu'] = to_array(self.__Tu, max_length)
        self.__parameters['VR'] = to_array(self.get_velocity_ratio(), max_length)
        self.__parameters['BR'] = to_array(self.__BR, max_length)
        self.__parameters['BR_normal'] = to_array(self.__BR * np.sin(np.radians(self.__alpha)), max_length)
        self.__parameters['BR_perpendicular'] = to_array(self.__BR * np.sin(np.radians(self.__alpha)), max_length)
        self.__parameters['DR'] = to_array(self.__DR, max_length)
        self.__parameters['IR'] = to_array(self.__IR, max_length)
        self.__parameters['IR_eff'] = to_array(self.__IR / (AR * AR), max_length)
        self.__parameters['IR_normal'] = to_array(self.__IR * np.sin(np.radians(self.__alpha)), max_length)
        self.__parameters['IR_perpendicular'] = to_array(self.__IR * np.sin(np.radians(self.__beta)), max_length)
        self.__parameters['ER'] = to_array(self.__get_er(), max_length)
        self.__parameters['TR'] = to_array(self.__TR, max_length)
        self.__parameters['Single_hole'] = to_array(1 if self.__is_single_hole else -1, max_length)
        self.__parameters['x_D'] = to_array(self.__x_D, max_length)
        self.__parameters['eff'] = to_array(self.__eff, max_length)

    def __heat_capacity_ratio(self):
        """Returns the coolant to mainstream isobaric heat capacity (cp) ratio"""
        cp_coolant = CoolProp.PropsSI("Cp0mass", "T", self.__Tc, "D", self.__rhoc, self.__coolant)
        cp_mainstream = CoolProp.PropsSI("Cp0mass", "T", self.__Tinf, "D", self.__rhoinf, self.__mainstream)

        return cp_coolant / cp_mainstream

    def __get_er(self):
        """Returns the advective capacity ratio, defined as blowing ratio * (ratio of specific heat capacities)"""
        return self.__BR * self.__heat_capacity_ratio()

    def __viscosity_ratio(self):
        """Returns the coolant to mainstream flow kinematic viscosity ratio"""

        mu_coolant = CoolProp.PropsSI("viscosity", "T", self.__Tc, "D", self.__rhoc, self.__coolant)
        mu_mainstream = CoolProp.PropsSI("viscosity", "T", self.__Tinf, "D", self.__rhoinf, self.__mainstream)

        return mu_coolant / mu_mainstream

    def __speed_of_sound_ratio(self):
        """Returns the coolant to mainstream speed of sound ratio"""
        # TODO: Should be probably field if Ac and Ainf are fields
        return self.__Ac / self.__Ainf

    def get_velocity_ratio(self):
        """Returns the coolant to mainstream flow velocity ratio"""
        return self.__BR / self.__DR

    def __get_single_correlation(self, AR, P_D, W_P, BR, DR, Tu, alpha, edge_offset, x_D):
        """Returns the appropriate correlation values for a set of parameters and x/D values"""
        if type(AR) is np.ndarray or \
                type(P_D) is np.ndarray or \
                type(W_P) is np.ndarray or \
                type(BR) is np.ndarray or \
                type(DR) is np.ndarray or \
                type(Tu) is np.ndarray or \
                type(alpha) is np.ndarray or \
                type(edge_offset) is np.ndarray or \
                type(x_D) is list:
            raise ValueError("For a single correlation, all inputs should be single values")
        if math.isclose(AR, 1.0):
            # Cylindrical hole, use Baldauf's correlation
            return correlations.baldauf(x_D, alpha, P_D, BR, DR, Tu / 100.0, b_0="fit")
        else:
            # Shaped hole, use Colban's correlation
            return correlations.colban(x_D - edge_offset, P_D, W_P, BR, AR)

    def get_correlations(self):
        """
        Get curves of film effectiveness-x/D as calculated by the appropriate correlation
        Returns
        -------
        tuple
            Tuple of lists of x/D and film effectiveness distributions
        """
        params = ['AR', 'P_D', 'W_P', 'BR', 'DR', 'Tu', 'Alpha', 'edge_offset']
        features = [self.__parameters[param] for param in params]
        features.append(self.__parameters['x_D'])
        no_dist = len(next(iter(features)))

        correlations = ([], [])
        for i in range(no_dist):
            next_feats = [feat[i] for feat in features]
            eff = self.__get_single_correlation(*next_feats)
            correlations[0].append(self.__parameters['x_D'][i])
            correlations[1].append(eff)

        return correlations

    def __transform_x(self, transformation: str):
        """
        Returns a transformed version of x/D
        Parameters
        ----------
        transformation : {"log", "reciprocal}
            Type of transformation, see generate_dataset() for more details
        Returns
        -------
        list
            List of lists of transformed x/D values
        """
        if transformation == "log":
            return [np.log(x) for x in self.__parameters['x_D']]
        elif transformation == "reciprocal":
            return [ER / (P_D * x)
                    for x, ER, P_D in zip(self.__parameters['x_D'], self.__parameters['ER'], self.__parameters['P_D'])]
        else:
            raise ValueError(f"Invalid x_D transformation {str}, valid values are: \"log\", \"reciprocal\"")

    def __get_single_feature_label_map(self, flow_params: Sequence[float],
                                       x_D: np.ndarray,
                                       eff: np.ndarray) -> Tuple[Sequence[float], Sequence[float]]:
        """Generates a feature-label tuple for a single set of flow parameters"""
        # Use list of features instead of parameters?
        # Flow parameters should be a single value, while x_D and eff are ndarrays
        if any(type(flow_param) is np.ndarray or type(flow_param) is list for flow_param in flow_params) or \
                type(x_D) is list or \
                type(eff) is list:
            raise ValueError("For a single feature label map, all features should be single values")
        return np.asarray([[*flow_params, curr_x] for curr_x in x_D]), eff

    def get_feature_label_maps(self, flow_param_list: Sequence[str] = None, include_correlations: bool = False,
                               data_filter: str = None, x_norm: str = None):
        """
        Returns a list of feature - label sets, one list per x/D - film effectiveness distribution
        Inclusion of flow parameters can be controlled, by default AR, W/D, Re, Ma and VR is included

        Parameters
        ----------
        flow_param_list: Sequence[str]
            List of flow parameters to include as features, in string format (case-insensitive).
            Possible values with aliases:
            Area ratio: "AR", "Area ratio"
            W/D: "W_D", "W/D", "Coverage ratio"
            Sin(beta): "Beta", "Orientation angle", "Compound angle"
            Re: "Re", "Reynolds", "Reynolds number"
            Ma: "Ma", "Mach", "Mach number"
            Tu: "Tu", "Turbulence intensity", "Turbulence number"
            VR: "VR", "Velocity ratio"
            BR: "BR", "Blowing ratio"
            DR: "DR", "Density ratio"
            IR: "IR", "Momentum flux ratio"
            Single hole: "Single", "Single hole", "Is single hole"
        include_correlations : bool, optional
            Controls whether correlation values are included, see generate_dataset() for details
        x_norm : {"log", "reciprocal"}, optional
            Transformation of x/D values, see generate_dataset() for details
        data_filter : {"cylindrical", "shaped"}, optional
            Filters datasets, see generate_dataset() for details
        Returns
        -------
        feats, labels:
            Tuple of list of ndarrays, with feats containing the features and labels containing the corresponding labels

        """
        if flow_param_list is None:
            flow_param_list = ["AR", "W/D", "Re", "Ma", "VR"]

        params = [Figure.to_param(x) for x in flow_param_list]
        is_shaped = [not math.isclose(ar, 1.0) for ar in self.__parameters['AR']]

        x_D_normalised = self.__parameters['x_D']
        if x_norm is not None:
            x_D_normalised = self.__transform_x(x_norm)

        features = [self.__parameters[param] for param in params]
        features.append(x_D_normalised)
        features.append(self.__parameters['eff'])

        no_dist = len(self.__parameters['x_D'])
        if not all(len(f) == no_dist for f in features):
            raise ValueError("Feature lists should all have length equal to number of distributions")

        feat_label_map = ([], [])
        for i in range(no_dist):
            next_feats = [feat[i] for feat in features]
            single_map_feats, single_map_labels = self.__get_single_feature_label_map(next_feats[:-2], *next_feats[-2:])
            feat_label_map[0].append(single_map_feats)
            feat_label_map[1].append(single_map_labels)

        if include_correlations:
            _, corr_eff = self.get_correlations()
            for i in range(len(feat_label_map[1])):
                feat_label_map[1][i] = np.stack((feat_label_map[1][i], corr_eff[i]), axis=1)

        if data_filter == "shaped":
            feat_label_map = [x for i, x in enumerate(feat_label_map[0]) if is_shaped[i]], \
                             [x for i, x in enumerate(feat_label_map[1]) if is_shaped[i]]
        elif data_filter == "cylindrical":
            feat_label_map = [x for i, x in enumerate(feat_label_map[0]) if not is_shaped[i]], \
                             [x for i, x in enumerate(feat_label_map[1]) if not is_shaped[i]]
        return feat_label_map

    def get_reynolds(self):
        """Returns the coolant Reynolds number"""
        return self.__Reinf * self.__BR / self.__viscosity_ratio()

    def get_mach(self):
        """Returns the coolant Mach number"""
        return self.__Mainf * self.get_velocity_ratio() / self.__speed_of_sound_ratio()

    def get_eff_uncertainty(self) -> float:
        """
        Returns the typical absolute measurement uncertainty in film effectiveness

        Returns
        -------
        uncertainty_eff_abs: float
            The typical measurement uncertainty in film effectiveness
        """
        return self.__uncertainty_eff_abs

    def get_example_count(self) -> int:
        """Returns number of examples in Figure"""
        return sum(len(x) for x in self.__parameters['eff'])

    def get_file(self) -> Path:
        """Returns Path to the corresponding JSON file"""
        return self.__file

    def __str__(self):
        return f"Study: {self.__ref}, {self.__fig}, varies parameters: {self.__variations_in}"

    @staticmethod
    def feature_names(flow_param_list: Sequence[str], x_norm: str = None) -> list:
        """Returns human-readable names of flow parameters"""
        names = [Figure.__PARAM_TO_READABLE[x] for x in flow_param_list]
        names.append(f"Horizontal distance{' (normalisation: ' + x_norm if x_norm is not None else ''}")
        return names

    @staticmethod
    def to_param(alias: str) -> str:
        """Converts an alias of a flow parameter to the proper parameter name"""
        return Figure.__ALIAS_TO_PARAM[alias.lower()]
