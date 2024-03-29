"""General utility functions."""

import numpy as np
import torch

class CustomStandardScaler():

    def __init__(self):
        self.mean = 0
        self.std = 0
    def fit(self, x: torch.Tensor, dim=None):
        self.mean = x.mean(dim=dim, keepdim=True)
        self.std = x.std(dim=dim, unbiased=False, keepdim=True)

    def transform(self, x: torch.Tensor):
        x -= self.mean
        x /= (self.std + 1e-10)
        return x

    def fit_transform(self, x: torch.Tensor, dim = None):
        self.fit(x, dim)
        return self.transform(x)

    def inverse(self, x_scaled: torch.Tensor):
        return x_scaled * (self.std + 1e-10) + self.mean


class CustomMinMaxScaler():

    def __init__(self):
        self.min = None
        self.max = None
        self.range = None

    def fit(self, x: torch.Tensor, dim=None):
        self.min = x.min(dim=dim, keepdim=True)[0]
        self.max = x.max(dim=dim, keepdim=True)[0]
        self.range = self.max - self.min

    def transform(self, x: torch.Tensor):
        return (x - self.min) / self.range

    def fit_transform(self, x: torch.Tensor, dim=None):
        self.fit(x, dim=dim)
        return self.transform(x)

    def inverse(self, x_scaled: torch.Tensor):
        return x_scaled * self.range + self.min


def get_geometry(phi, psi, Lphi_D, Lpsi_D, alpha):
    """Evaluate geometric parameters from diffusion angles and lengths.

    This assumes that the shaped hole edges are rounded with the same radius as
    a cylindrical metering section.

    Parameters
    ----------
    phi : float
        Lateral divergence half angle [degrees]
    psi : float
        Forward divergence full angle [degrees]
    Lphi_D : float
        Lateral divergence length, normalised by hole diameter [-]
    Lpsi_D : float
        Forward divergence length, normalised by hole diameter [-]
    alpha : float
        Inclination angle [degrees]

    Returns
    -------
    AR : float
        Exit-to-inlet area ratio for fully-enclosed cross sections, i.e. not
        all the way up to the hole axis. See Schroeder and Thole (2014) for a
        diagram. [-]
    x_D_edge : float
        Streamwise location of the hole trailing edge from hole axis,
        normalised by the hole diameter [-]
    W_D : float
        Lateral hole width on the hole axis, normalised by hole diameter [-]
    """

    # Tangents of angles
    tan_alpha = np.tan(np.radians(alpha))
    tan_phi = np.tan(np.radians(phi))
    tan_psi = np.tan(np.radians(psi))
    sin_alpha = np.sin(np.radians(alpha))
    cos_alpha = np.cos(np.radians(alpha))

    # Move back to fully-enclosed cross section
    Lphi_D_prime = Lphi_D - 0.5 / tan_alpha
    Lpsi_D_prime = Lpsi_D - 0.5 / tan_alpha

    # Evaluate area ratio
    AR = 1.0 + 4.0 / np.pi * (
        2.0 * Lphi_D_prime * tan_phi
        + Lpsi_D_prime * tan_psi
        + 2.0 * Lphi_D_prime * tan_phi * Lpsi_D_prime * tan_psi
    )

    # Coordinates at start of forwards divergence
    dy_D_start = Lpsi_D * sin_alpha + 0.5 * cos_alpha
    dx_D_start = Lpsi_D * cos_alpha - 0.5 * sin_alpha

    # Streamwise length of forwards divergence
    dx_D_diverge = dy_D_start / np.tan(np.radians(alpha - psi))

    # Evaluate distance from axis to trailing edge
    x_D_edge = dx_D_diverge - dx_D_start

    # Evaluate width
    W_D = 1.0 + 2.0 * Lphi_D * tan_phi

    return AR, x_D_edge, W_D


def trailing_edge_offset(alpha: float, psi: float, Lpsi_D: float) -> float:
    """

    Returns the horizontal distance of the trailing edge from the center of the hole

    Parameters
    ----------
    alpha : float
        Inclination angle [degrees]
    psi : float
        Forward divergence full angle [degrees]
    Lpsi_D : float
        Forward divergence length, normalised by hole diameter [-]

    Returns
    -------
    offset_D: float
        The horizontal distance between the trailing edge and the intersection of the horizontal plane
    and the centerline of the hole, normalised by hole diameter [-]
    """

    # This is a completely unnecessary reimplementation of x_D_edge from get_geometry() because
    # I didn't notice it existed and assumed the third return value was Lphi*/D
    # Remarkably though, the two are equal save for a 1e-15 order difference, probably due to floating point arithmetic
    sin_alpha = np.sin(np.radians(alpha))
    tan_alpha = np.tan(np.radians(alpha))
    tan_psi = np.tan(np.radians(psi))

    # Distance between centerline and the theoretical edge of the hole if psi was 0
    x1_D = 1 / (2 * sin_alpha)

    # Distance between the actual trailing edge the theoretical edge of the hole if psi was 0
    x2_D = (1 / (2 * tan_alpha) + Lpsi_D) * tan_psi * np.sin(np.radians(90 + psi)) / np.sin(np.radians(alpha - psi))
    offset_D = x1_D + x2_D
    return offset_D
