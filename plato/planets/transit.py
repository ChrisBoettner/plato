import numpy as np
from astropy import units as u
from astropy import constants as const


class TransitModel:
    def __init__(self) -> None:
        pass

    def calculate_impact_parameter(
        self,
        a: u.Quantity[u.AU],
        r_star: u.Quantity[u.Rsun],
        cos_i: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculate the impact parameter for a given set of parameters.

        Parameters
        ----------
        a : u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.
        r_star : u.Quantity[u.Rsun]
            Radius of the star, in solar radii.
        cos_i : float | np.ndarray
            Cosine of the inclination angle of the planet's orbit.

        Returns
        -------
        float | np.ndarray
            Impact parameter.
        """

        return (a / r_star).decompose().value * cos_i

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
        porb: u.Quantity[u.day],
        r_p: u.Quantity[u.Rearth],
        r_star: u.Quantity[u.Rsun],
        m_star: u.Quantity[u.Msun],
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
        u1 : float | np.ndarray
            First limb darkening coefficient.
        u2 : float | np.ndarray
            Second limb darkening coefficient.

        Returns
        -------
        float | np.ndarray
            Transit depth.
        """

        a = self.calculate_semimajor_axis(porb, m_star)
        radius_ratio = (r_p / r_star).decompose()

        return radius_ratio**2 * self.calculate_limb_darkening_correction(
            a, r_star, cos_i, u1, u2
        )

    def calculate_limb_darkening_correction(
        self,
        a: u.Quantity[u.AU],
        r_star: u.Quantity[u.Rsun],
        cos_i: float | np.ndarray,
        u1: float | np.ndarray,
        u2: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculate the limb darkening correction for a given set of parameters,
        following Heller et al. (2019), using the quadratic limb darkening law.
        For parameter values where the planet does not transit the star, the
        correction factor is set to zero.

        Parameters
        ----------
        a : u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.
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
            Limb darkening correction factor.

        """
        cos_i = np.asarray(cos_i)
        u1 = np.asarray(u1)
        u2 = np.asarray(u2)

        b = self.calculate_impact_parameter(a, r_star, cos_i)
        b = np.asarray(b)

        mask = b < 1
        corr_factor = np.full(a.shape, 0)

        coeff = np.sqrt(1 - b[mask] ** 2)
        I_p = 1 - u1[mask] * (1 - coeff) - u2[mask] * (1 - coeff) ** 2
        I_A = 1 - u1[mask] / 3 - u2[mask] / 6

        corr_factor[mask] = I_p / I_A
        return corr_factor
