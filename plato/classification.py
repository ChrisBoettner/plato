import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from galpy.orbit import Orbit
from typing import Optional
from scipy.stats import norm


def calculate_galactic_velocities(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Galactic velocities U, V, and W in the heliocentric frame
    using galpy. The input dataframe should contain the following columns:
        - ra: Right ascension in degrees
        - dec: Declination in degrees
        - parallax: Parallax in mas
        - pmra: Proper motion in RA in mas/yr
        - pmdec: Proper motion in Dec in mas/yr
        - radial_velocity: Radial velocity in km/s


    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the Galactic velocities
        U, V, and W in the heliocentric frame in km/s.

    """
    for key in ["ra", "dec", "parallax", "pmra", "pmdec", "radial_velocity"]:
        if key not in dataframe.columns:
            raise ValueError(f"Dataframe must contain {key!r}.")

    coord = SkyCoord(
        ra=dataframe["ra"].to_numpy() * u.degree,
        dec=dataframe["dec"].to_numpy() * u.degree,
        distance=(1 / dataframe["parallax"].to_numpy()) * u.pc,
        pm_ra_cosdec=dataframe["pmra"].to_numpy() * u.mas / u.yr,
        pm_dec=dataframe["pmdec"].to_numpy() * u.mas / u.yr,
        radial_velocity=dataframe["radial_velocity"].to_numpy() * u.km / u.s,
    )

    orbits = Orbit(
        vxvv=[
            coord.ra,
            coord.dec,
            coord.distance,
            coord.pm_ra_cosdec,
            coord.pm_dec,
            coord.radial_velocity,
        ],
        radec=True,
    )

    return orbits.U(), orbits.V(), orbits.W()


def component_probability(
    U: float | np.ndarray,
    V: float | np.ndarray,
    W: float | np.ndarray,
    parameter: dict[str, float],
    log: bool = False,
) -> float | np.ndarray:
    """
    Calculate the probability of a star with velocities
    U, V, and W to belong to a given component characterized
    by the parameters sigma_U, sigma_V, sigma_W, and V_asym.

    Parameters
    ----------
    U : float | np.ndarray
        The Galactic velocity U in km/s.
    V : float | np.ndarray
        The Galactic velocity V in km/s.
    W : float | np.ndarray
        The Galactic velocity W in km/s.
    parameter : dict[str, float]
        A dictionary containing the parameters of the component.
        The dict must contain:
            - sigma_U: The velocity dispersion in U in km/s.
            - sigma_V: The velocity dispersion in V in km/s.
            - sigma_W: The velocity dispersion in W in km/s.
            - V_asym: The mean velocity in V in km/s.

    log : bool, optional
        If True, return the log of the probability instead,
        by default False

    Returns
    -------
    float | np.ndarray
        The probability of the star to belong to the component.

    """
    for key in ["sigma_U", "sigma_V", "sigma_W", "V_asym"]:
        if key not in parameter.keys():
            raise ValueError(f"Parameter dict must contain {key!r}.")
    if log:
        U_prob = norm.logpdf(U, loc=0, scale=parameter["sigma_U"])
        V_prob = norm.logpdf(V, loc=parameter["V_asym"], scale=parameter["sigma_V"])
        W_prob = norm.logpdf(W, loc=0, scale=parameter["sigma_W"])
        return U_prob + V_prob + W_prob

    U_prob = norm.pdf(U, loc=0, scale=parameter["sigma_U"])
    V_prob = norm.pdf(V, loc=parameter["V_asym"], scale=parameter["sigma_V"])
    W_prob = norm.pdf(W, loc=0, scale=parameter["sigma_W"])
    return U_prob * V_prob * W_prob


def relative_probability(
    U: float | np.ndarray,
    V: float | np.ndarray,
    W: float | np.ndarray,
    component_1: str | dict[str, float],
    component_2: str | dict[str, float],
    component_dict: Optional[dict[str, dict[str, float]]] = None,
) -> float | np.ndarray:
    """
    Calculate the relative probability of a star to belong to
    component 1 with respect to component 2. The probability is
    calculated as the ratio of the probability of the star to belong
    to component 1 and the probability of the star to belong to component 2,
    weighted by the relative fraction of the components.

    The components must be either provided by strings found in the
    component_dict or by stand-alone dictionaries with the necessary
    parameters.

    If component_dict is None, the default components are:
    - thin disk: X=0.94, sigma_U=35, sigma_V=20, sigma_W=16, V_asym=-15
    - thick disk: X=0.06, sigma_U=67, sigma_V=38, sigma_W=35, V_asym=-46
    - halo: X=0.0015, sigma_U=160, sigma_V=90, sigma_W=90, V_asym=-220

    Parameters
    ----------
    U : float | np.ndarray
        The Galactic velocity U in km/s.
    V : float | np.ndarray
        The Galactic velocity V in km/s.
    W : float | np.ndarray
        The Galactic velocity W in km/s.
    component_1 : str | dict[str, float]
        The first component, either a string matching the keys in the
        component_dict or a dictionary with the parameters.
    component_2 : str | dict[str, float]
        The first component, either a string matching the keys in the
        component_dict or a dictionary with the parameters.
    component_dict : Optional[dict[str, dict[str, float]]], optional
        A dictionary containing the parameters of the components.
        Must be of the form
        {component: {X, sigma_U, sigma_V, sigma_W, V_asym}}.
        By default None. If None, the default components are used.

    Returns
    -------
    float | np.ndarray
        The relative probability of the star to belong to component 1
        with respect to component 2.

    """
    if component_dict is None:
        component_dict = {
            "thin disk": {
                "X": 0.94,
                "sigma_U": 35,
                "sigma_V": 20,
                "sigma_W": 16,
                "V_asym": -15,
            },
            "thick disk": {
                "X": 0.06,
                "sigma_U": 67,
                "sigma_V": 38,
                "sigma_W": 35,
                "V_asym": -46,
            },
            "halo": {
                "X": 0.0015,
                "sigma_U": 160,
                "sigma_V": 90,
                "sigma_W": 90,
                "V_asym": -220,
            },
        }

    components = [component_1, component_2]
    for i, component in enumerate(components):
        if isinstance(component, str):
            if component not in component_dict.keys():
                raise ValueError(
                    f"Component {component!r} not found in component_dict,"
                    f"with keys {list(component_dict.keys())}."
                )
            else:
                components[i] = component_dict[component]

        elif isinstance(component, dict):
            for key in ["X", "sigma_U", "sigma_V", "sigma_W", "V_asym"]:
                if key not in component.keys():
                    raise ValueError(
                        f"Parameter dict {component!r} must contain {key!r}."
                    )
        else:
            raise ValueError(
                "Component must be a string or a dictionary with the parameters."
            )

    component_1, component_2 = components
    assert isinstance(component_1, dict)
    assert isinstance(component_2, dict)

    prob_1 = component_probability(U, V, W, component_1)
    prob_2 = component_probability(U, V, W, component_2)
    return component_1["X"] * prob_1 / (component_2["X"] * prob_2)


def classify_stars(
    dataframe: pd.DataFrame,
    lower_cut: float = 0.1,
    upper_cut: float = 10,
    overwrite: bool = False,
    raise_errors: bool = True,
) -> pd.DataFrame:
    """
    Classify stars into different populations based on their Galactic
    velocities. A new column 'Population' is added to the dataframe,
    containing the classification. The classification is based on
    the relative probabilities of the stars to belong to the thin disk,
    thick disk, and halo components.

    The input dataframe should contain the following columns:
        - ra: Right ascension in degrees
        - dec: Declination in degrees
        - parallax: Parallax in mas
        - pmra: Proper motion in RA in mas/yr
        - pmdec: Proper motion in Dec in mas/yr
        - radial_velocity: Radial velocity in km/s

    The classes are defined as follows:
        - Thin Disk
        - Thick Disk Candidate
        - Thick Disk
        - Halo Candidate
        - Halo

    The classification is based on the relative probabilities
    of the stars to belong to the thin disk, thick disk, and
    halo components. Specifically, the the stars are classified
    to a component if the relative probability is below a lower
    cut or above an upper cut. For in-between values, the stars
    are classified as candidates.

    For halo stars and halo candidates, the stars are checked
    if they are classified as thick disk beforehand. If so, an
    error is raised. Deactivate this check by setting raise_errors
    to False.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.
    lower_cut : float, optional
        The lower cut for the relative probability, if
        e.g. the relative probabiltiy thick disk/thin
        disk is below this value, the star is classified
        as thin disk, by default 0.1.
    upper_cut : float, optional
        The upper cut for the relative probability, if
        e.g. the relative probabiltiy thick disk/thin
        disk is above this value, the star is classified
        as thick disk, by default 10.
    overwrite : bool, optional
        If True, overwrite the 'Population' column if it
        already exists, by default False.
    raise_errors : bool, optional
        If True, raise errors if some stars are not classified
        or if some halo stars or halo candidates are classified
        as thick disk beforehand, by default True.

    Returns
    -------
    pd.DataFrame
        The input dataframe with a new column 'Population'
        containing the classification.

    Raises
    ------
    ValueError
        If the dataframe already contains a 'Population' column
        and overwrite is False.
    RuntimeError
        If some some halo stars or halo candidates
        are not classified as thick disk beforehand.
    RuntimeError
        If some stars are not classified.
    """

    if "Population" in dataframe.columns and not overwrite:
        raise ValueError(
            "Dataframe already contains a 'Population' column, "
            "use overwrite=True to overwrite."
        )

    new_df = dataframe.copy()

    # calculate Galactic velocities
    u, v, w = calculate_galactic_velocities(dataframe)

    # calculate relative probabilities between thick disk and thin disk
    td_d = relative_probability(u, v, w, "thick disk", "thin disk")
    # calculate relative probabilities between thick disk and halo
    h_td = relative_probability(u, v, w, "halo", "thick disk")

    # classify stars
    populations = {
        "Thin Disk": (td_d < lower_cut),
        "Thick Disk Candidate": ((td_d > lower_cut) & (td_d < upper_cut)),
        "Thick Disk": (td_d > upper_cut),
        "Halo Candidate": ((h_td > lower_cut) & (h_td < upper_cut)),
        "Halo": (h_td > upper_cut),
    }

    new_df["Population"] = "Unknown"
    for population, mask in populations.items():
        # Halo and Halo Candidate stars should be subset
        # of Thick Disk Candidate stars that were already classified
        if population in ["Halo Candidate", "Halo"]:
            if (
                {"Thick Disk", "Halo Candidate"}
                < set(new_df["Population"][mask].unique())
            ) and raise_errors:
                raise RuntimeError(
                    "Some halo stars or halo candidates are not classified "
                    "as thick disk beforehand."
                )
        new_df.loc[mask, "Population"] = population

    # check if all stars are classified
    if (new_df["Population"] == "Unknown").any() and raise_errors:
        raise RuntimeError("Some stars are not classified.")

    return new_df
