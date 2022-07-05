"""General utility functions."""

import numpy as np

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
