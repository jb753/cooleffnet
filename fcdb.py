import itertools
import json
import random
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import CoolProp.CoolProp as CoolProp

import util


class CoolingDatabase:

    def __init__(self, database_dir: Path, verbose: bool=False):
        self.__datafiles = [f for f in database_dir.iterdir() if f.is_file() and f.suffix == ".json"]
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
                         flow_param_list: Sequence[str] = None):
        if flow_param_list is None:
            flow_param_list = ["AR", "W/D", "Re", "Ma", "VR"]

        if self.__verbose:
            print("Generating dataset...")
            print(f"Wanted features: {Figure.feature_names(flow_param_list)}")

        self.__training_feats = []
        self.__training_labels = []
        self.__training_files = []
        self.__test_feats = []
        self.__test_labels = []
        self.__test_files = []

        file_list = self.__datafiles.copy()
        if shuffle:
            random.shuffle(file_list)

        unique_vals = set()
        train_count = 0
        test_count = 0
        for file in file_list:
            fig = Figure(file)
            feats, labels = fig.get_feature_label_maps(flow_param_list)
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
                    self.__training_files.append(file)
                elif test_count < test_min:
                    self.__test_feats += feats
                    self.__test_labels += labels
                    self.__test_files.append(file)
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
        if not self.__dataset_generated:
            raise RuntimeError("Requested training dataset without calling generate_dataset() before")
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

        # FIXME: Potentially throwing away data...
        feat_sets = feat_sets[:no_splits]
        label_sets = label_sets[:no_splits]

        lengths = [sum(len(y) for y in x) for x in feat_sets]
        if padding == "max":
            largest_set_size = max(lengths)
            for feat_set, label_set, length in zip(feat_sets, label_sets, lengths):
                padding_length = largest_set_size - length
                if padding_length > 0:
                    feat_set.append(np.zeros((padding_length, feat_set[0].shape[-1])))
                    label_set.append(np.zeros(padding_length))

        feat_sets = [np.concatenate(x) for x in feat_sets]
        label_sets = [np.atleast_2d(np.concatenate(x)).T for x in label_sets]

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

    def get_crossvalidation_sets(self, no_splits: int = 5, padding: str = None):
        if not self.__dataset_generated:
            raise RuntimeError("Requested cross-validation dataset without calling generate_dataset() before")
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
    def get_dataset(self, test: bool = False, normalize_features = False, normalize_labels=False,
                     zero_mean_features=False, zero_mean_labels=False, return_stats: bool = False):
        if not self.__dataset_generated:
            raise RuntimeError("Requested training dataset without calling generate_dataset() before")

        feat_matrix = np.concatenate(self.__training_feats) if not test else np.concatenate(self.__test_feats)
        label_matrix = np.concatenate(self.__training_labels) if not test else np.concatenate(self.__test_labels)
        label_matrix = np.atleast_2d(label_matrix).T

        feat_means, feat_stdevs = np.mean(feat_matrix, axis=0), np.std(feat_matrix, axis=0)
        label_means, label_stdevs = np.mean(label_matrix, axis=0), np.std(label_matrix, axis=0)
        if zero_mean_features or normalize_features:
            feat_matrix -= feat_means
        if zero_mean_labels or normalize_labels:
            label_matrix -= label_means

        if normalize_features:
            feat_matrix /= feat_stdevs
        if normalize_labels:
            label_matrix /= label_stdevs

        # stats = feat_means, feat_stdevs, label_means, label_stdevs
        stats = {
            'feat_means': feat_means,
            'feat_stdevs': feat_stdevs,
            'label_means': label_means,
            'label_stdevs': label_stdevs
        }

        if return_stats:
            return feat_matrix, label_matrix, stats
        else:
            return feat_matrix, label_matrix

    def get_files(self, test: bool = False):
        if not self.__dataset_generated:
            raise RuntimeError("Requested training dataset without calling generate_dataset() before")
        return self.__test_files if test else self.__training_files


class Figure:

    def __init__(self, file: Path):
        """Initialises a Figure object from a JSON figure file"""

        with open(file) as figure_file:
            self.__figure_dict = json.load(figure_file)

            self.__variations_in = []
            # Convert every list to a numpy array
            for top_key in self.__figure_dict.keys():
                for key, value in self.__figure_dict[top_key].items():
                    if type(value) is list and type(value) is not str:
                        if type(value[0]) is list:
                            # Assume that if first element is a list, then it's a list of lists
                            # If list of lists, turn it to a list of NumPy arrays
                            self.__figure_dict[top_key][key] = [np.asarray(x) for x in value]
                        else:
                            if top_key != 'distributions':
                                self.__variations_in.append(key)
                            self.__figure_dict[top_key][key] = np.asarray(self.__figure_dict[top_key][key])

            self.__is_single_hole = self.__figure_dict['geometry']['is_single_hole']
            self.__alpha = self.__figure_dict['geometry']['alpha']
            self.__beta = self.__figure_dict['geometry']['beta']
            self.__phi = self.__figure_dict['geometry']['phi']
            self.__psi = self.__figure_dict['geometry']['psi']
            self.__Lphi_D = self.__figure_dict['geometry']['Lphi_D']
            self.__Lpsi_D = self.__figure_dict['geometry']['Lpsi_D']
            self.__is_x_origin_trailing_edge = self.__figure_dict['geometry']['is_x_origin_trailing_edge']

            # Check if Vinf exists
            self.__Vinf = self.__figure_dict['dimensional']['Vinf'] if 'Vinf' in self.__figure_dict['dimensional'] else None
            self.__Tc = self.__figure_dict['dimensional']['Tc'] if 'Tc' in self.__figure_dict['dimensional'] else None
            self.__Tinf = self.__figure_dict['dimensional']['Tinf'] if 'Tinf' in self.__figure_dict['dimensional'] else None
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
            self.__IR = self.__BR * self.__BR / self.__DR # Momentum flux ratio
            # Apparently universal gas constant can be different for different fluids in CoolProp
            # Not very universal anymore, now is it?
            self.__Rc = CoolProp.PropsSI("gas_constant", self.__coolant) / self.__Mc
            self.__Rinf = CoolProp.PropsSI("gas_constant", self.__mainstream) / self.__Minf

            # Assuming ideal gas
            self.__TR = self.__MR / self.__DR # Coolant to mainstream temperature ratio
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

    def __get_single_feature_label_map(self, flow_params: Sequence[float],
                                       x_D: np.ndarray,
                                       eff: np.ndarray) -> Tuple[Sequence[float], Sequence[float]]:
        # Use list of features instead of parameters?
        # Flow parameters should be a single value, while x_D and eff are ndarrays
        if any(type(flow_param) is np.ndarray or type(flow_param) is list for flow_param in flow_params) or \
                type(x_D) is list or \
                type(eff) is list:
            raise ValueError("For a single feature label map, all features should be single values")
        return np.asarray([[*flow_params,curr_x] for curr_x in x_D]), eff

    def get_feature_label_maps(self, flow_param_list: Sequence[str] = None):
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
            Sin(beta): "Beta", "Orientation angle", "Compound angle
            Re: "Re", "Reynolds", "Reynolds number"
            Ma: "Ma", "Mach", "Mach number"
            Tu: "Tu", "Turbulence intensity", "Turbulence number"
            VR: "VR", "Velocity ratio"
            BR: "BR", "Blowing ratio"
            DR: "DR", "Density ratio"
            IR: "IR", "Momentum flux ratio"

        Returns
        -------
        feats, labels:
            Tuple of list of ndarrays, with feats containing the features and labels containing the corresponding labels

        """
        if flow_param_list is None:
            flow_param_list = ["AR", "W/D", "Re", "Ma", "VR"]

        AR, _, W_D = util.get_geometry(self.__phi, self.__psi, self.__Lphi_D, self.__Lpsi_D, self.__alpha)
        Beta = np.sin(np.radians(self.__beta))
        Re = self.get_reynolds()
        Ma = self.get_mach()
        Tu = self.__Tu
        VR = self.get_velocity_ratio()
        BR = self.__BR
        DR = self.__DR
        IR = self.__IR
        x_D = self.__x_D
        eff = self.__eff

        # Usings sets could technically be faster, but this should still be okay
        features = []
        if any(param_string.lower() in ["ar", "area ratio"] for param_string in flow_param_list):
            features.append(AR)
        if any(param_string.lower() in ["w/d", "w_d", "coverage ratio"] for param_string in flow_param_list ):
            features.append(W_D)
        if any(param_string.lower() in ["beta", "compound angle", "orientation angle"] for param_string in flow_param_list ):
            features.append(Beta)
        if any(param_string.lower() in ["re", "reynolds", "reynolds number"] for param_string in flow_param_list ):
            features.append(Re)
        if any(param_string.lower() in ["ma", "mach", "mach number"] for param_string in flow_param_list ):
            features.append(Ma)
        if any(param_string.lower() in ["tu", "turbulence intensity", "tu number"] for param_string in flow_param_list ):
            features.append(Tu)
        if any(param_string.lower() in ["vr", "velocity ratio"] for param_string in flow_param_list ):
            features.append(VR)
        if any(param_string.lower() in ["br", "blowing ratio"] for param_string in flow_param_list ):
            features.append(BR)
        if any(param_string.lower() in ["dr", "density ratio"] for param_string in flow_param_list ):
            features.append(DR)
        if any(param_string.lower() in ["ir", "momentum flux ratio"] for param_string in flow_param_list ):
            features.append(IR)

        # Keep x_D and eff at the end in any case
        features.append(x_D)
        features.append(eff)
        is_list = [False] * len(features)
        is_list[:-2] = [type(x) is np.ndarray for x in features[:-2]]
        is_list[-2:] = [type(x) is list for x in features[-2:]]

        # Look up if there is a neater way of doing this, but for now it'll do
        feat_label_map = ([], [])
        if type(eff) is list or type(x_D) is list:
            # List so, multiple result sets
            length = len(features[next(i for i, x in enumerate(is_list) if x)])
            for feature, is_a_list in zip(features, is_list):
                if is_a_list and len(feature) != length:
                    raise ValueError("Feature lists should have equal length")

            for i in range(length):
                next_feats = [feat[i] if is_list[j] else feat for j, feat in enumerate(features)]
                single_map_feats, single_map_labels = self.__get_single_feature_label_map(next_feats[:-2], *next_feats[-2:])
                feat_label_map[0].append(single_map_feats)
                feat_label_map[1].append(single_map_labels)
        else:
            # TODO: Add more checks
            # Maybe move this to get_single_feature_label_map()?
            # Sanity check:
            if len(eff) != len(x_D):
                raise ValueError("Arrays of x and y coordinates should have same length")
            single_map_feats, single_map_labels = self.__get_single_feature_label_map(features[:-2], x_D, eff)
            feat_label_map = [single_map_feats], [single_map_labels]

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

    def __str__(self):
        return f"Study: {self.__ref}, {self.__fig}, varies parameters: {self.__variations_in}"

    @staticmethod
    def feature_names(flow_param_list: Sequence[str] = None) -> list:
        name_list = []
        if flow_param_list is None:
            return ["Area ratio", "Coverage ratio", "Reynolds number", "Mach number", "Velocity ratio", "Horizontal position over diameter"]
        else:
            if any(param_string.lower() in ["ar", "area ratio"] for param_string in flow_param_list ):
                name_list.append("Area ratio")
            if any(param_string.lower() in ["w/d", "w_d", "coverage ratio"] for param_string in flow_param_list ):
                name_list.append("Coverage ratio")
            if any(param_string.lower() in ["beta", "compound angle", "orientation angle"] for param_string in flow_param_list):
                name_list.append("Compound angle")
            if any(param_string.lower() in ["re", "reynolds", "reynolds number"] for param_string in flow_param_list ):
                name_list.append("Reynolds number")
            if any(param_string.lower() in ["ma", "mach", "mach number"] for param_string in flow_param_list ):
                name_list.append("Mach number")
            if any(param_string.lower() in ["tu", "turbulence intensity", "tu number"] for param_string in flow_param_list ):
                name_list.append("Tu number")
            if any(param_string.lower() in ["vr", "velocity ratio"] for param_string in flow_param_list ):
                name_list.append("Velocity ratio")
            if any(param_string.lower() in ["br", "blowing ratio"] for param_string in flow_param_list ):
                name_list.append("Blowing ratio")
            if any(param_string.lower() in ["dr", "density ratio"] for param_string in flow_param_list ):
                name_list.append("Density ratio")
            if any(param_string.lower() in ["ir", "momentum flux ratio"] for param_string in flow_param_list ):
                name_list.append("Momentum flux ratio")

            name_list.append("Horizontal position over diameter")
            return name_list