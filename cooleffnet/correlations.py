"""Benchmark correlations to compare to data-driven methods

Baldauf et al. (2002) - cylindrical holes.
Colban et al. (2011) - cylindrical holes.

"""

import numpy as np
import io
import pkgutil

# Load the fit data
Baldauf_b0_coeffs = np.loadtxt(io.StringIO(pkgutil.get_data(__name__, "models/Baldauf_b0_coeffs.dat").decode("utf-8")))
"""Fitted polynomial surface to optimum b0 for Baldauf et al. (2002) data.

There is a typo in Eqn. (31) from their paper, so that the value of b0 does
not agree with what they quote in the worked example from the appendix. We
repair the correlation by fitting a polynomial surface as a replacement for
Eqn. (31),
    b_0(x,y) = A + Bx + Cy + Dxy + Ex^2 + Fy^2 + ...
where the independent variables are x = P_D, y = sin(alpha), and A, B, ...
are constants to be found by minimising RMS error between fitted b_0 and
the optimum b_0 (that minimises RMS effectiveness error in each case). Some
experimentation shows that optimum b_0 must be clipped to a unit interval
to remove outliers.

We save the coefficients in a file in the same directory as this script,
and read in when the module is imported.

There is no improvement in accuracy for polynomial order > 2.
"""


def _colban(x_D, P_D, W_P, BR, AR, validate=False):
    """Film effectiveness correlation for shaped holes, Colban et al. (2011).

    Colban, W. F., Thole, K. A., and Bogard, D. (2010).
    "A Film-Cooling Correlation for Shaped Holes on a Flat-Plate Surface"
    *J. Turbomach.* 133(1) pp. 011002.
    https://doi.org/10.1115/1.4002064

    Parameters
    ----------
    x_D : float
        Streamwise distance downstream of hole trailing edge, normalised by
        hole metering section diameter.
    P_D : float
        Lateral hole pitch normalised by hole metering section diameter.
    W_P : float
        Coverage ratio - hole breakout width over lateral hole pitch.
    BR : float
        Hole blowing ratio - coolant mass flux over mainstream mass flux.
    AR : float
        Hole area ratio - outlet over inlet cross sectional areas. Note
        defined for fully-enclosed parts of the hole, see Fig. 2 in the paper.
    validate : bool, default False
        If true, then return NaN when outside the correlation limits.

    Returns
    -------
    eff : float
        Laterally-averaged film effectiveness at each streamwise location as
        calculated from their correlation, Eqn. (19) in the paper.
    """

    # Assemble data for validity limits
    # BR,  Coverage ratio, xifac
    lim = np.array(
        [[0.2, 2.5], [0.31, 0.65], [0.17, 1.17]]
    )
    xifac = AR / BR / P_D
    val = np.array((BR, W_P, xifac))

    # Return NaN if any variable outside limits from Table 4
    is_invalid = np.any(np.logical_or(val < lim[:, 0], val > lim[:, 1]))
    if (is_invalid and validate) or x_D < 0.0:
        return np.nan

    # Constants from Table 3
    C1 = 0.1721
    C2 = -0.2664
    C3 = 0.8749

    # Non-dimensional distance, Eqn. (18)
    xi = 4.0 / np.pi * x_D * P_D / BR / AR

    # Correlation Eqn. (19)
    eff = 1.0 / (1.0 / W_P + C1 * (BR ** C2) * (xi ** C3))

    return eff


def _baldauf(
    x_D, alpha, P_D, BR, DR, Tu, b_0=None, validate=False, verbose=False
):
    """Film effectiveness correlation for cylinders, Baldauf et al. (2002).

    Baldauf , S., Scheurlen, M., Schulz , A., and Wittig, S. (2002).
    "Correlation of Film-Cooling Effectiveness From Thermographic Measurements
    at Enginelike Conditions"
    J. Turbomach. 124(4) pp. 686–698
    https://doi.org/10.1115/1.1504443

    Parameters
    ----------
    x_D : float
        Streamwise distance downstream of hole *axis*, normalised by
        hole metering diameter.
    alpha : float
        Hole inclination angle in degrees.
    P_D : float
        Lateral hole pitch normalised by hole metering section diameter.
    BR : float
        Hole blowing ratio, coolant mass flux over main-stream mass flux.
    DR : float
        Hole density ratio, coolant density over main-stream density.
    Tu : float
        Main-stream turbulence intensity, not in %
    b_0 : str or float
        If a scalar float, override the corellation to enforce a value of b_0.
        If string 'paper', use incorrect Eqn. (31) from the paper.
        If string 'fit', use polynomial fit for optimal b_0.
        Default value of None uses the fit option.
    verbose : bool, default False
        If true, then print intermediate variables for verification.

    Returns
    -------
    eff : float
        Laterally-average film effectiveness at the streamwise location as
        calculated from their correlation.
    """

    # Hole parameters
    VR = BR / DR

    # Constants for the base curve fit, Table 2
    xi_0 = 9.0
    eta_0 = 5.8
    a_star = 4.0
    b_star = 0.7
    c_star = 0.24

    # Convert angles to radians
    alphar = np.radians(alpha)

    # Turbulence correction parameters Eqns. (34), (35)
    b_star_T = 0.7 * (
        1.0
        + (
            1.22 / (1.0 + 7.0 * (P_D - 1.0) ** -7.0)
            + 0.87
            + np.cos(2.5 * alphar)
        )
        * np.exp(2.6 * Tu - 0.0012 / Tu ** 2.0 - 1.76)
    )
    eta_0_T = 2.5 * (eta_0 / 2.5) ** (b_star_T / b_star)

    # Transformation to a specific maximum
    # Eqns. (11) - (15)
    a = 0.2
    b = np.exp(1.92 - 7.5 * P_D ** -1.5)
    c = 0.7 + 336.0 * np.exp(-1.85 * P_D)
    mu_0 = 0.125 + 0.063 * P_D ** 1.8
    eta_c0 = 0.465 / (1.0 + 0.048 * P_D ** 2.0)

    # Eqn. (9)
    mu = VR * DR ** 0.8 * (1.0 - (0.03 + 0.11 * (5.0 - P_D)) * np.cos(alphar))

    # Eqn. (18)
    xi_c = 0.6 + 0.4 * (2.0 - np.cos(alphar)) / (
        1.0 + ((P_D - 1.0) / 3.3) ** 6.0
    )

    # Adjacent jet interaction effects on maximum
    # Eqns. (22) - (27)
    xi_hat = (
        1.17
        * (1.0 - (P_D - 1.0) / (1.0 + 0.2 * (P_D - 1.0) ** 2.0))
        * (np.cos(2.3 * alphar) + 2.45)
    )
    eta_hat = 0.022 * (P_D + 1.0) * (0.9 - np.sin(2.0 * alphar)) - (
        0.08 + 0.46 / (1.0 + (P_D - 3.2) ** 2.0)
    )
    g = 0.75 * (1.0 - np.exp(-0.8 * (P_D - 1.0)))
    k = (
        2.0 * (1.0 - np.exp(0.57 * (1.0 - P_D)))
        + 0.91 * np.cos(alphar) ** 0.65
    )
    eta_s = 1.0 + eta_hat / (1.0 + (VR / k * DR ** g) ** -5.0)
    xi_s = 1.0 + xi_hat / (1.0 + (VR / k * DR ** g) ** -5.0)

    # Adjacent jet interaction effects downstream
    # Eqns. (29) - (33)
    a_1 = 0.04 + 0.23 * P_D + (0.95 - 0.19 * P_D) * np.cos(1.5 * alphar)
    if b_0 == "paper":
        b_0 = (
            0.8
            - 0.014 * (P_D ** 2.0)
            + (
                (1.5 - (2.0 / np.sqrt(P_D)))
                * np.sin(
                    0.86 * alphar * (1.0 + 0.754 / (1.0 + 0.87 * (P_D ** 2.0)))
                )
            )
        )
    elif b_0 is None or b_0 == "fit":
        b_0 = np.polynomial.polynomial.polyval2d(
            P_D, np.sin(alphar), Baldauf_b0_coeffs
        )

    b_1 = b_0 / (1.0 + BR ** -3.0)
    c_1 = 7.5 + P_D
    xi_1 = 65.0 / ((BR / 2.5) ** a_1)

    # Determine transformed distances for desired x/D
    # by inverting Eqns. (40)
    xi_prime = x_D * P_D * xi_c * 4.0 / np.pi / (VR ** ((P_D / 3.0) ** -0.75))

    # Turbulence-dependent base curve, Eqn. (36)
    xi_n0 = xi_prime / xi_0
    apb_c_star = (a_star + b_star_T) * c_star
    denominator = (1.0 + xi_n0 ** apb_c_star) ** (1.0 / c_star)
    eta_star_prime = eta_0_T * xi_n0 ** a_star / denominator

    # Account for adjacent jet interaction, Eqn. (37)
    xi_n1 = xi_prime / xi_1
    eta_star = (
        0.1
        * (eta_star_prime / 0.1) ** (1.0 / eta_s)
        * (1.0 + xi_n1 ** (b_1 * c_1)) ** (1.0 / c_1)
    )

    # Characteristic peak effectiveness, Eqn. (38)
    mu_n = mu / mu_0
    apbc = (a + b) * c
    eta_c = eta_c0 * eta_star * mu_n ** a / (1.0 + mu_n ** apbc) ** (1.0 / c)

    # Specific peak effectivness, Eqn. (39)
    eff = eta_c * DR ** (0.9 / P_D) / np.sin(alphar) ** (0.06 * P_D)

    # Print details
    if verbose:
        print("Base curve: \nb*T = %.8f, eta0T = %.8f" % (b_star_T, eta_0_T))
        print("Transformation to a specific maximum:")
        print(
            "xi_c = %.8f, eta_c0 = %.8f\nmu = %.8f, mu_0 = %.8f"
            % (xi_c, eta_c0, mu, mu_0)
        )
        print("a = %.8f, b = %.8f, c = %.8f" % (a, b, c))
        print("Adjacent jet interaction effects on maximum:")
        print(
            """xi_hat = %.8f, eta_hat = %.8f
            g = %.8f, k = %.8f\nxi_s = %.8f, eta_s = %.8f"""
            % (xi_hat, eta_hat, g, k, xi_s, eta_s)
        )
        print("Adjacent jet interaction effects downstream:")
        print(
            "xi_1 = %.8f, a_1 = %.8f\nb_0 = %.8f, b_1 = %.8f, c_1 = %.8f"
            % (xi_1, a_1, b_0, b_1, c_1)
        )

    return eff


# Make numpy-friendly
colban = np.vectorize(_colban)
baldauf = np.vectorize(_baldauf)


def polyfit2d(x, y, z, order):
    """Fit a two-dimensional polynomial surface with given maximum order.

    After https://stackoverflow.com/questions/33964913/

    Parameters
    ----------
    x, y : array-like
        1-D vectors for the independent variables at each data point.
    z : array-like
        1-D vector for the dependent variable at each data point.
    order : int
        The maximum power of the polynomial terms.

    Returns
    -------
    coeff : array
        Vector of fitted polynomial coefficients, evaluate using,
        z = np.polynomial.polynomial.polyval2d(x, y, coeffs)
    """

    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((order + 1, order + 1))
    # solve array
    A = np.zeros((coeffs.size, x.size))
    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x ** i * y ** j
        A[index] = arr
    # do leastsq fitting and return leastsq result
    return (
        np.linalg.lstsq(A.T, z, rcond=None)[0]
        .reshape((order + 1, order + 1))
        .T
    )
