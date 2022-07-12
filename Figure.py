import json
from pathlib import Path
import itertools

import numpy as np
import CoolProp.CoolProp as CoolProp

import util


class Figure:

    def __init__(self, file: Path):
        """Initialises a Figure object from a JSON figure file"""

        with open(file) as figure_file:
            self.__figure_dict = json.load(figure_file)

            # Convert every list to a numpy array
            for top_key in self.__figure_dict.keys():
                for key, value in self.__figure_dict[top_key].items():
                    if type(value) is list and type(value) is not str:
                        if type(value[0]) is list:
                            # Assume that if first element is a list, then it's a list of lists
                            # If list of lists, turn it to a list of NumPy arrays
                            self.__figure_dict[top_key][key] = [np.asarray(x) for x in value]
                        else:
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


    def __get_single_feature_label_map(self, AR: float, W_D: float, Re: float, Ma: float, VR: float, x_D: np.ndarray,
                                       eff: np.ndarray) -> list:
        # Use list of features instead of parameters?
        # AR, W_D, Re, Ma, VR should be a single value, while x_D and eff are ndarrays
        if type(AR) is np.ndarray or \
                type(W_D) is np.ndarray or \
                type(Re) is np.ndarray or \
                type(Ma) is np.ndarray or \
                type(VR) is np.ndarray or \
                type(x_D) is list or \
                type(eff) is list:
            raise ValueError("For a single feature label map, all features should be single values")
        return [([AR, W_D, Re, Ma, VR, curr_x], curr_eff) for curr_x, curr_eff in zip(x_D, eff)]

    def get_feature_label_maps(self):

        AR, _, W_D = util.get_geometry(self.__phi, self.__psi, self.__Lphi_D, self.__Lpsi_D, self.__alpha)
        Re = self.get_reynolds()
        Ma = self.get_mach()
        VR = self.get_velocity_ratio()
        x_D = self.__x_D
        eff = self.__eff

        # Keep x_D and eff at the end in any case
        features = [AR, W_D, Re, Ma, VR, x_D, eff]
        is_list = [False] * len(features)
        is_list[:-2] = [type(x) is np.ndarray for x in features[:-2]]
        is_list[-2:] = [type(x) is list for x in features[-2:]]

        # Look up if there is a neater way of doing this, but for now it'll do
        feat_label_map = []
        if type(eff) is list or type(x_D) is list:
            # List so, multiple result sets
            length = len(features[next(i for i, x in enumerate(is_list) if x)])
            for feature, is_a_list in zip(features, is_list):
                if is_a_list and len(feature) != length:
                    raise ValueError("Feature lists should have equal length")

            for i in range(length):
                next_feats = [feat[i] if is_list[j] else feat for j, feat in enumerate(features)]
                feat_label_map.append(self.__get_single_feature_label_map(*next_feats))
        else:
            # TODO: Add more checks
            # Maybe move this to get_single_feature_label_map()?
            # Sanity check:
            if len(eff) != len(x_D):
                raise ValueError("Arrays of x and y coordinates should have same length")
            feat_label_map = self.__get_single_feature_label_map(AR, W_D, Re, Ma, VR, x_D, eff)

        return feat_label_map

    def get_reynolds(self):
        """Returns the coolant Reynolds number"""
        return self.__Reinf * self.__BR / self.__viscosity_ratio()

    def get_mach(self):
        """Returns the coolant Mach number"""
        return self.__Mainf * self.get_velocity_ratio() / self.__speed_of_sound_ratio()