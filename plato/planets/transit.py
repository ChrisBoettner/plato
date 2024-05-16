import numpy as np
from astropy import units as u
from astropy import constants as const


class TransitModel:
    def __init__(self) -> None:
        pass

    def calculate_transit_duration(
        self,
        porb: u.Quantity[u.day],
        r_p: u.Quantity[u.Rearth],
        r_star: u.Quantity[u.Rsun],
        m_star: u.Quantity[u.Msun],
        cos_i: float | np.ndarray,
    ) -> u.Quantity[u.hour]:
        """
        Calculate the transit duration for a given set of parameters,
        following Eq. 3 from Seager & Mallén-Ornelas (2003) assuming
        a circular orbit.

        Parameters
        ----------
        porb : u.Quantity[u.day]
            Orbital period of the planet, in days.
        r_p : u.Quantity[u.Rearth]
            Radius of the planet, in Earth radii.
        r_star : u.Quantity[u.Rsun]
            Radius of the star, in solar radii.
        m_star : u.Quantity[u.Msun]
            Mass of the star, in solar masses.
        cos_i : float | np.ndarray
            Cosine of the inclination angle of the planet's orbit.

        Returns
        -------
        u.Quantity[u.hour]
            Transit duration, in hours.

        """
        a = self.calculate_semimajor_axis(porb, m_star)

        term1 = (r_p / r_star).decompose().value ** 2
        term2 = (a / r_star).decompose().value * cos_i

        inner_term = np.clip((1 + term1) ** 2 - term2**2, 0, None)
        outer_term = np.where(
            np.abs(cos_i) < 1,
            (r_star / a).decompose().value * np.sqrt(inner_term / (1 - cos_i**2)),
            0,
        )

        t_transit = porb / np.pi * np.arcsin(outer_term)

        return t_transit.to(u.hour)

    def calculate_semimajor_axis(
        self,
        porb: u.Quantity[u.day],
        m_star: u.Quantity[u.Msun],
    ) -> u.Quantity[u.AU]:
        """
        Calculate the semimajor axis of the planet's orbit, following
        Kepler's third law. Planet mass is assumed to be negligible.

        Parameters
        ----------
        porb : u.Quantity[u.day]
            Orbital period of the planet, in days.
        m_star : u.Quantity[u.Msun]
            Mass of the star, in solar masses.

        Returns
        -------
        u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.

        """
        numerator = porb**2 * const.G * m_star  # type: ignore
        denominator = 4 * np.pi**2

        return ((numerator / denominator) ** (1 / 3)).to(u.AU)

    def calculate_transit_depth(
        self,
        r_p: u.Quantity[u.Rearth],
        r_star: u.Quantity[u.Rsun],
        cos_i: float | np.ndarray,
        u1: float | np.ndarray,
        u2: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculate the transit depth for a given set of parameters,
        following Eq. 4 from Heller et al. (2019) using the quadratic
        limb darkening law.

        Parameters
        ----------
        r_p : u.Quantity[u.Rearth]
            Radius of the planet, in Earth radii.
        r_star : u.Quantity[u.Rsun]
            Radius of the star, in solar radii.
        cos_i : float | np.ndarray
            Cosine of the inclination angle of the planet's orbit.
        u1 : float | np.ndarray
            First limb darkening coefficient.
        u2 : float | np.ndarray
            Second limb darkening coefficient.

        Returns
        -------
        float | np.ndarray
            Transit depth.
        """

        radius_ratio = (r_p / r_star).decompose()

        return radius_ratio**2 * self.calculate_limb_darkening_correction(cos_i, u1, u2)

    def calculate_limb_darkening_correction(
        self,
        cos_i: float | np.ndarray,
        u1: float | np.ndarray,
        u2: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculate the limb darkening correction for a given set of parameters,
        following Heller et al. (2019), using the quadratic limb darkening law.

        Parameters
        ----------
        cos_i : float | np.ndarray
            Cosine of the inclination angle of the planet's orbit.
        u1 : float | np.ndarray
            First limb darkening coefficient.
        u2 : float | np.ndarray
            Second limb darkening coefficient.

        Returns
        -------
        float | np.ndarray
            Limb darkening correction factor.

        """
        cos_gamma = np.sqrt(1 - cos_i**2)  # gamma = pi/2 - i, and using trig identity

        I_p = 1 - u1 * (1 - cos_gamma) - u2 * (1 - cos_gamma) ** 2
        I_A = 1 - u1 / 3 - u2 / 6
        return I_p / I_A
