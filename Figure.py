import json
from pathlib import Path
import itertools

import numpy as np
import CoolProp.CoolProp as CoolProp

import util


class Figure:

    def __init__(self, file: Path):
        """Initialises a Figure object from a JSON figure file"""
        #print(file.name)

        with open(file) as figure_file:
            self.__figure_dict = json.load(figure_file)
            #print(self.__figure_dict)

            # Convert every list to a numpy array
            for top_key in self.__figure_dict.keys():
                for key, value in self.__figure_dict[top_key].items():
                    if type(value) is list and type(value) is not str:
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
                self.__x_D += util.trailing_edge_offset(self.__alpha, self.__psi, self.__Lpsi_D)

            # Fill out None values:
            M_coolant = CoolProp.PropsSI("molar_mass", self.__coolant)
            M_inf = CoolProp.PropsSI("molar_mass", self.__mainstream)

            self.__MR = M_coolant / M_inf # Molar mass ratio

            # Assuming ideal gas
            self.__TR = self.__MR / self.__DR # Coolant to mainstream temperature ratio
            if self.__Tc is None:
                self.__Tc = self.__Tinf * self.__TR
            if self.__Tinf is None:
                self.__Tinf = self.__Tc / self.__TR

            # FIXME: Still need a proper pressure value from somewhere...
            self.__p = 1e5
            a_mainstream = CoolProp.PropsSI("speed_of_sound", "T", self.__Tinf, "P", self.__p, self.__mainstream)
            if self.__Mainf is None:
                self.__Mainf = self.__Vinf / a_mainstream

            if self.__Vinf is None:
                self.__Vinf = self.__Mainf * a_mainstream

    def __viscosity_ratio(self):
        """Returns the coolant to mainstream flow kinematic viscosity ratio"""

        p = 1e5 # ***PLACEHOLDER***

        mu_coolant = CoolProp.PropsSI("viscosity", "T", self.__Tc, "P", p, self.__coolant)
        mu_mainstream = CoolProp.PropsSI("viscosity", "T", self.__Tinf, "P", p, self.__mainstream)

        return mu_coolant / mu_mainstream

    def __speed_of_sound_ratio(self):
        """Returns the coolant to mainstream speed of sound ratio"""
        p = 1e5 # ***PLACEHOLDER***

        a_coolant = CoolProp.PropsSI("speed_of_sound", "T", self.__Tc, "P", p, self.__coolant)
        a_mainstream = CoolProp.PropsSI("speed_of_sound", "T", self.__Tinf, "P", p, self.__mainstream)

        return a_coolant / a_mainstream

    def get_velocity_ratio(self):
        """Returns the coolant to mainstream flow velocity ratio"""

        # Check if iterable to allow for numpy arrays an others?
        # Check if magic simplification exists with itertools (can convert to numpy array and go from there?)

        return self.__BR / self.__DR
        # if type(self.__BR) is list:
        #     if type(self.__DR) is list:
        #         return [x / y for x, y in zip(self.__BR / self.__DR)]
        #     else:
        #         return [x / self.__DR for x in self.__BR]
        # else:
        #     if type(self.__DR) is list:
        #         return [self.__BR / y for x, y in zip(self.__BR / self.__DR)]
        #     else:
        #         return self.__BR / self.__DR

    def get_features(self):
        """Returns a list of features ready to be used in the regression step"""
        # Should VR be added as well?
        return [self.get_reynolds(), self.get_mach()]

    def get_labels(self):
        """Returns a list of labels to be used in the regression step"""
        return [self.__x_D, self.__eff]

    def get_reynolds(self):
        """Returns the coolant Reynolds number"""
        # if type(self.__BR) is list:
        #     return [self.__Reinf * BR/ self.__viscosity_ratio() for BR in self.__BR]
        return self.__Reinf * self.__BR / self.__viscosity_ratio()

    def get_mach(self):
        """Returns the coolant Mach number"""
        # VR = self.get_velocity_ratio()
        # if type(VR) is list:
        #     return [self.__Mainf * vr / self.__speed_of_sound_ratio() for vr in VR]
        return self.__Mainf * self.get_velocity_ratio() / self.__speed_of_sound_ratio()


if __name__ == "__main__":
    data_files = [path for path in Path("data").iterdir() if path.is_file() and path.suffix == ".json"]

    # TODO: Add proper testing
    data_set_no = 0
    for file in data_files:
        # FIXME: Either make object more versatile or change McNamara datasets
        if "McNamara" in file.name:
            continue
        test = Figure(file)
        print(test.get_velocity_ratio())
        print(test.get_features())
        print(test.get_reynolds())
        print(test.get_mach())
        if type(test.get_reynolds()) is float:
            data_set_no += 1
        else:
            data_set_no += len(test.get_reynolds())

