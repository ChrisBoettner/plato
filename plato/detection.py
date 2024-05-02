from typing import Optional

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.stats import gamma


class NoiseModel:
    """
    A class to model the noise in the PLATO mission, following
    the description in Boerner+2022.
    """

    def __init__(
        self,
        n_photoelectrons_ref: int = 177100,
        n_photoelectrons_brightest_pixel_ref: int = 47400,
        n_images: int = 144,
        n_mask: float = 9.5,
        ccd_readout_noise_factor: float = 44.3,
        fee_readout_noise_factor: float = 37.0,
        asd_jitter: float = 0.54e-6,
        observation_frequency: int = 278,
        signal_spread_factor: float = 0.27,
        background_rate: float = 60,
        stray_light_rate: float = 64,
        contaminating_star_rate: float = 0.01,
        exposure_time: int = 21,
        smearing_rate: float = 45,
        charge_transfer_time: float = 4,
        psf_breathing_noise_factor: float = 20e-6,
        fee_offset_sensitivity: float = 1,
        adc_gain: float = 25,
        fee_temp_stability: float = 1.0,
        fee_temp_knowledge: float = 0.01,
        fee_offset_converstion: float = 0.66,
    ) -> None:
        """
        Initializes the noise model with reference parameters.

        Parameters
        ----------
        n_photoelectrons_ref : int, optional
            Reference number of photoelectrons in the entire mas,
            by default 177100.
        n_photoelectrons_brightest_pixel_ref : int, optional
            Reference number of photoelectrons in the brightest pixel,
            by default 47400.
        n_images : int,
            Number of images taken within 1 hour,
            by default 144 (cadence of 25s, with 21s
            exposure time and 4s signal transfer time).
        n_mask : float, optional
            Effective number of pixels in the mask,
            by default 9.5.
        ccd_readout_noise_factor : float, optional
            CCD readout noise factor, by default 44.3.
        fee_readout_noise_factor : float, optional
            FEE readout noise factor, by default 37.0.
        asd_jitter : float, optional
            ASD (amplitude spectral density) jitter factor, by default 0.54e-6.
        observation_frequency : int, optional
            Observation frequency in microHz, by default 278.
        signal_spread_factor : float, optional
            Fraction of energy in the brightest pixel,
            by defaults 0.27.
        background_rate : float, optional
            Background photoelectron rate per pixel per second,
            by default 60.
        stray_light_rate : float, optional
            Stray light photoelectron rate per pixel per second,
            by default 64.
        contaminating_star_rate : float, optional
            Contaminating star signal as a fraction of target star signal,
            by default 0.01.
        exposure_time : int, optional
            Exposure time in seconds, by default 21.
        smearing_rate : float, optional
            Median photoelectron rate for smearing noise,
            by default 45.
        charge_transfer_time : float, optional
            Charge transfer time in seconds, by default 4.
        psf_breathing_noise_factor : float, optional
            PSF breathing noise factor in ppm on camera level,
            by default 20e-6.
        fee_offset_sensitivity : float, optional
            FEE offset sensitivity to temperature in ADU/K,
            by default 1.
        adc_gain : float, optional
            ADC gain factor, by default 25.
        fee_temp_stability : float, optional
            FEE temperature stability in K, by default 1.0.
        fee_temp_knowledge : float, optional
            FEE temperature knowledge in K, by default 0.01.
        fee_offset_converstion : float, optional
            FEE offset conversion factor, by default 0.66.
        """
        self.n_photoelectrons_ref = n_photoelectrons_ref
        self.n_photoelectrons_brightest_pixel_ref = n_photoelectrons_brightest_pixel_ref
        self.n_images = n_images
        self.n_mask = n_mask
        self.ccd_readout_noise_factor = ccd_readout_noise_factor
        self.fee_readout_noise_factor = fee_readout_noise_factor
        self.asd_jitter = asd_jitter
        self.observation_frequency = observation_frequency
        self.signal_spread_factor = signal_spread_factor
        self.background_rate = background_rate
        self.stray_light_rate = stray_light_rate
        self.contaminating_star_rate = contaminating_star_rate
        self.exposure_time = exposure_time
        self.smearing_rate = smearing_rate
        self.charge_transfer_time = charge_transfer_time
        self.psf_breathing_noise_factor = psf_breathing_noise_factor
        self.fee_offset_sensitivity = fee_offset_sensitivity
        self.adc_gain = adc_gain
        self.fee_temp_stability = fee_temp_stability
        self.fee_temp_knowledge = fee_temp_knowledge
        self.fee_offset_converstion = fee_offset_converstion

    def calculate_flux(self, magnitude_v: float) -> float:
        """
        Calculates the flux of a star given its magnitude.

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.

        Returns
        -------
        float
            The flux of the star.
        """
        return 10 ** (-0.4 * (magnitude_v - 11))

    def calculate_n_photoelectrons(self, magnitude_v: float) -> float:
        """
        Calculates the number of photoelectrons in the entire mask, based
        on the V band magnitude of the star. (Very approximate, WIP)

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.

        Returns
        -------
        float
            The number of photoelectrons in the entire mask.
        """
        return self.calculate_flux(magnitude_v) * self.n_photoelectrons_ref

    def calculate_n_photoelectrons_brightest_pixel(self, magnitude_v: float) -> float:
        """
        Calculates the number of photoelectrons in the brightest pixel, based
        on the V band magnitude of the star. (Very approximate, WIP)

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.

        Returns
        -------
        float
            The number of photoelectrons in the brightest pixel.
        """
        return (
            self.calculate_flux(magnitude_v) * self.n_photoelectrons_brightest_pixel_ref
        )

    def ccd_readout_noise(self, magnitude_v: float, n_cameras: int) -> float:
        """
        Calculates the CCD readout noise for a star given its magnitude.

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float
            The CCD readout noise.
        """
        return (
            self.ccd_readout_noise_factor
            / self.calculate_n_photoelectrons(magnitude_v)
            * np.sqrt(self.n_mask / n_cameras / self.n_images)
        )

    def fee_readout_noise(self, magnitude_v: float, n_cameras: int) -> float:
        """
        Calculates the FEE readout noise for a star given its magnitude.

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float
            The FEE readout noise.
        """
        return (
            self.fee_readout_noise_factor
            / self.calculate_n_photoelectrons(magnitude_v)
            * np.sqrt(self.n_mask / n_cameras / self.n_images)
        )

    def jitter_noise(self) -> float:
        """
        Calculates the jitter noise.

        Returns
        -------
        float
            The jitter noise.
        """
        return self.asd_jitter * np.sqrt(self.observation_frequency)

    def misc_photon_noise(self, magnitude_v: float, n_cameras: int) -> float:
        """
        Calculates the miscellaneous photon noise, consistent of
        background noise, stray light noise, and contaminating star noise.

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float
            The miscellaneous photon noise.
        """
        background_noise = self.background_rate * self.exposure_time
        stray_light_noise = self.stray_light_rate * self.exposure_time
        contaminating_star_noise = (
            self.contaminating_star_rate
            * self.calculate_n_photoelectrons_brightest_pixel(magnitude_v)
        )
        return (
            np.sqrt(background_noise + stray_light_noise + contaminating_star_noise)
            / self.calculate_n_photoelectrons_brightest_pixel(magnitude_v)
            * self.signal_spread_factor
            * np.sqrt(self.n_mask / n_cameras / self.n_images)
        )

    def ccd_smearing_noise(self, magnitude_v: float, n_cameras: int) -> float:
        """
        Calculates the CCD smearing noise.

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float
            The CCD smearing noise.
        """
        return (
            np.sqrt(self.smearing_rate * self.charge_transfer_time)
            / self.calculate_n_photoelectrons_brightest_pixel(magnitude_v)
            * self.signal_spread_factor
            * np.sqrt(self.n_mask / n_cameras / self.n_images)
        )

    def psf_breathing_noise(self, n_cameras: int) -> float:
        """
        Calculates the PSF breathing noise.

        Parameters
        ----------
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float
            The PSF breathing noise.
        """
        return self.psf_breathing_noise_factor / np.sqrt(n_cameras * self.n_images)

    def fee_offset_stability_noise(self, magnitude_v: float, n_cameras: int) -> float:
        """
        Calculates the FEE offset stability noise.

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float
            The FEE offset stability noise.
        """
        return (
            self.fee_offset_sensitivity
            * self.adc_gain
            * self.fee_temp_knowledge
            / self.calculate_n_photoelectrons_brightest_pixel(magnitude_v)
            * self.signal_spread_factor
            * self.fee_offset_converstion
            / np.sqrt(n_cameras * self.n_images)
        )

    def photon_noise(self, magnitude_v: float, n_cameras: int) -> float:
        """
        Calculates the photon noise.

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float
            The photon noise.
        """
        return 1 / np.sqrt(
            self.calculate_n_photoelectrons(magnitude_v) * n_cameras * self.n_images
        )

    def random_noise(self, magnitude_v: float, n_cameras: int) -> float:
        """
        Calculates the random noise, which is the (Pythagorean) sum of the CCD readout
        noise, FEE readout noise, miscellaneous photon noise, and CCD
        smearing noise.

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float
            The random noise.
        """
        return np.sqrt(
            self.ccd_readout_noise(magnitude_v, n_cameras) ** 2
            + self.fee_readout_noise(magnitude_v, n_cameras) ** 2
            + self.misc_photon_noise(magnitude_v, n_cameras) ** 2
            + self.ccd_smearing_noise(magnitude_v, n_cameras) ** 2
        )

    def systematic_noise(self, magnitude_v: float, n_cameras: int) -> float:
        """
        Calculates the systematic noise, which is the (Pythagorean) sum of the jitter
        noise, PSF breathing noise, and FEE offset stability noise.

        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float
            The systematic noise.
        """
        return np.sqrt(
            self.jitter_noise() ** 2
            + self.psf_breathing_noise(n_cameras) ** 2
            + self.fee_offset_stability_noise(magnitude_v, n_cameras) ** 2
        )

    def calculate_NSR(
        self,
        magnitude_v: float,
        n_cameras: int,
        stellar_variability: float = 10e-6,
    ) -> float:
        """
        Calculate the noise-to-signal ratio.
        The NSR consists of photometric precision, with consits of
        photon noise, random noise, and systematic noise, and the
        stellar variability.
        The photometric precision is the signal-to-noise ratio
        for a constant source observed (and integrated) over one hour. The
        value depends on the magnitude of the star and the number of cameras
        observing the star.
        The stellar variability is the standard deviation of the variability
        of the star on timescales of ~1hr, assumed to be white noise. Correlations
        and trends are not considered.


        Parameters
        ----------
        magnitude_v : float
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float
            The noise-to-signal ratio.
        """

        photon_noise = self.photon_noise(magnitude_v, n_cameras)
        random_noise = self.random_noise(magnitude_v, n_cameras)
        systematic_noise = self.systematic_noise(magnitude_v, n_cameras)
        photometric_precision_squared = (
            photon_noise**2 + random_noise**2 + systematic_noise**2
        )

        return np.sqrt(photometric_precision_squared + stellar_variability**2)


class DetectionModel:
    """
    A class to calculate the signal-to-noise ratio (SNR) for
    a transiting exoplanet, as described in Matuszewski+2023
    (Section 4.2).
    """

    def __init__(self, cdpp_model: Optional[NoiseModel] = None) -> None:
        """
        Initializes the SNR model with a NoiseModel instance.

        Parameters
        ----------
        cdpp_model : Optional[CDPPModel], optional
            A CDPPModel instance, by default None, in which
            case a new CDPPModel instance will be created with
            default parameters.
        """
        if cdpp_model is None:
            self.noise_model = NoiseModel()
        else:
            self.noise_model = cdpp_model

    def calculate_transit_duration(
        self,
        porb: u.day,
        r_star: u.Rsun,
        a: u.AU,
    ) -> u.hour:
        """
        Calculates the duration of a transit. Added a
        correction factor to account for the finite size
        of the star, see Seager & Mallén-Ornelas (2003).

        Parameters
        ----------
        porb : u.day
            The orbital period of the planet.
        r_star : u.Rsun
            The radius of the host star.
        a : u.AU
            The semi-major axis of the planet's orbit.

        Returns
        -------
        u.hour
            The duration of the transit.
        """
        t0 = porb * r_star / (np.pi * a)
        correction_factor = (1 + r_star / a) ** 2
        return (t0 * correction_factor).to(u.hour)

    def calculate_semi_major_axis(
        self,
        porb: u.day,
        m_star: u.Msun,
    ) -> u.AU:
        """
        Calculates the semi-major axis of the planet's orbit
        using Kepler's Third Law.

        Parameters
        ----------
        porb : u.day
            The orbital period of the planet.
        m_star : u.Msun
            The mass of the host star.

        Returns
        -------
        u.AU
            The semi-major axis of the planet's orbit.
        """
        numerator = const.G * m_star * porb**2  # type: ignore
        denominator = 4 * np.pi**2
        return ((numerator / denominator) ** (1 / 3)).to(u.AU)

    def calculate_p_N_plus_1_transits(
        self,
        t_mission: u.year,
        porb: u.day,
    ) -> float:
        """
        Calculates the probability
        of having N+1 transits during the mission.

        Parameters
        ----------
        t_mission : u.year
            The duration of the mission.
        porb : u.day
            The orbital period of the planet.

        Returns
        -------
        float
            The probability
            of having N+1 transits.

        """
        remainder = (
            t_mission % porb
        )  # additional time after N transits that may lead to N+1 transits
        p_n_plus_one_transits = remainder / porb  # probability of N+1 transits
        return p_n_plus_one_transits.decompose().value

    def calculate_number_of_transits(
        self,
        t_mission: u.year,
        porb: u.day,
        return_average: bool = False,
    ) -> float:
        """
        Estimates the average number of transits during the mission,
        accounting for the timing of the first transit.
        If return_average is True, the number of transits is
        calculated as the weighted average of cases where
        the initial transit timing timing leads to N or N+1 transits.
        Otherwise N transits is returned (wher N is the guaranteed
        number of transits).

        Parameters
        ----------
        t_mission : u.year
            The duration of the mission.
        porb : u.day
            The orbital period of the planet.
        t_transit : u.hour
            The duration of a single transit.
        return_average : bool, optional
            If True, returns the average number of transits.
        Returns
        -------
        float
            The estimated number of transits.
        """

        n_transits = t_mission // porb  # number of N definitve transits

        if return_average:
            p_n_plus_one_transits = self.calculate_p_N_plus_1_transits(t_mission, porb)
            p_n_transits = 1 - p_n_plus_one_transits  # probability of N transits

            return p_n_plus_one_transits * (n_transits + 1) + p_n_transits * n_transits

        return n_transits.decompose().value

    def calculate_snr(
        self,
        r_planet: u.Rearth,
        r_star: u.Rsun,
        porb: u.day,
        m_star: u.Msun,
        magnitude_v: float,
        n_cameras: int,
        t_mission: u.year = 2 * u.year,
        stellar_variability: float = 10e-6,
        extra_transit: bool = False,
    ) -> float:
        """
        Calculates the signal-to-noise ratio (SNR) for a
        transiting exoplanet.

        Parameters
        ----------
        r_planet : u.Rearth
            The radius of the planet.
        r_star : u.Rsun
            The radius of the host star.
        porb : u.day
            The orbital period of the planet.
        m_star : u.Msun
            The mass of the host star.
        magnitude : float
            The apparent magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.
        t_mission : u.year
            The duration of the mission, by default 2 years.
        stellar_variability : float
            The standard deviation of the variability of the star
            on timescales of ~1hr, assumed to be white noise, by default 10e-6.
        extra_transit : bool
            If True, an additional transit is assumed to be observed,
            by default False. This is useful, since the estimated
            number of transits is the lower (guaranteed) number of transits,
            but depending on the timing of the first transit, an additional
            transit may be observed.

        Returns
        -------
        float
            The signal-to-noise ratio (SNR).
        """
        a = self.calculate_semi_major_axis(porb, m_star)
        t_transit = self.calculate_transit_duration(porb, r_star, a)
        n_transits = self.calculate_number_of_transits(t_mission, porb)

        if extra_transit:
            n_transits += 1

        transit_depth = (r_planet / r_star) ** 2

        # calculate noise-to-signal ratio for an observation of 1 hour
        noise_rate = (
            self.noise_model.calculate_NSR(
                magnitude_v,
                n_cameras,
                stellar_variability,
            )
            * u.hour**-0.5
        )

        # noise integrated over entire transit(s)
        noise = noise_rate / np.sqrt(t_transit.to(u.hour)) / np.sqrt(n_transits)

        # note:
        # noise is technically a noise-to-signal-ratio, not noise in itself,
        # but the transit depth "signal" should also be multiplied by the
        # signal (in terms of flux or photoelectrons), meaning that the
        # values cancel out in the SNR calculation.
        # The confusion potential confusion here stems from the fact that
        # in the noise model above, the "signal" corresponds to the measured
        # flux value, while in the detection model the "signal" corresponds to
        # the transit depth.

        snr = transit_depth / noise
        return snr.to(u.hour).value  # return SNR

    @staticmethod
    def linear_detection_efficiency(
        snr: float | np.ndarray,
        lower_threshold: float = 6,
        upper_threshold: float = 10,
    ) -> float | np.ndarray:
        """
        Calculates the detection efficiency as a function
        of the signal-to-noise ratio (SNR) using a linear model.
        The efficiency is a linear function of the SNR, with a lower
        threshold below which the efficiency is 0 and an upper threshold
        above which the efficiency is 1, from Fressin+2013.

        Parameters
        ----------
        snr : float
            The signal-to-noise ratio.
        lower_threshold : float, optional
            The lower threshold below which the detection
            efficiency is 0, by default 6 (Fressin+2013).
        upper_threshold : float, optional
            The upper threshold above which the detection
            efficiency is 1, by default 10 (taken from the
            Plato Red book, which assumes a 100% efficiency
            for an snr>10).

        Returns
        -------
        float
            The detection efficiency as a function of the SNR.
        """
        if snr < lower_threshold:
            return 0.0
        elif snr > upper_threshold:
            return 1.0
        else:
            return (snr - lower_threshold) / (upper_threshold - lower_threshold)

    @staticmethod
    def gamma_detection_efficiency(
        snr: float | np.ndarray,
        gamma_a: float = 30.87,
        gamma_b: float = 0.271,
        gamma_c: float = 0.940,
    ) -> float | np.ndarray:
        """
        Calculates the detection efficiency as a function
        of the signal-to-noise ratio (SNR) using a gamma distribution model.
        The efficiency is a gamma cumulative distribution function (CDF) of the SNR,
        from Christiansen+2016, Christiansen+2017.

        Parameters
        ----------
        snr : float | np.ndarray
            The signal-to-noise ratio.
        gamma_a : float, optional
            The shape parameter of the gamma distribution,
            by default 30.87 (Christiansen+2017).
        gamma_b : float, optional
            The scale parameter of the
            gamma distribution, by default 0.271 (Christiansen+2017).
        gamma_c : float, optional
            The overall normalisation parameter of the
            gamma distribution, by default 0.940 (Christiansen+2017).


        Returns
        -------
        float | np.ndarray
            The detection efficiency as a function of the SNR.
        """
        return gamma_c * gamma.cdf(snr, a=gamma_a, loc=0, scale=gamma_b)

    def detection_efficiency(
        self,
        r_planet: u.Rearth,
        r_star: u.Rsun,
        porb: u.day,
        m_star: u.Msun,
        magnitude_v: float,
        n_cameras: int,
        *,
        t_mission: u.year = 2 * u.year,
        stellar_variability: float = 10e-6,
        min_transits: int = 2,
        mode: str = "gamma",
        kwargs_detection_efficiency: Optional[dict] = None,
    ) -> float | np.ndarray:
        """
        Calculates the detection efficiency as a function
        of the signal-to-noise ratio (SNR). Two models are
        implemented:
        - A linear model, where the efficiency is a linear function
        of the SNR, with a lower threshold below which the efficiency
        is 0 and an upper threshold above which the efficiency is 1,
        from Fressin+2013.
        - A gamma model, where the efficiency is a gamma cumulative
        distribution function (CDF) of the SNR, from
        Christiansen+2016, Christiansen+2017.

        The detection efficiency is calculated for N and N+1 transits,
        and the final efficiency is a weighted average of the two
        efficiencies, where the weight is the probability of observing
        N+1 transits. If the number of transits is below a certain
        threshold (min_transits), the efficiency is set to 0.

        Parameters
        ----------
        r_planet : u.Rearth
            The radius of the planet.
        r_star : u.Rsun
            The radius of the host star.
        porb : u.day
            The orbital period of the planet.
        m_star : u.Msun
            The mass of the host star.
        magnitude : float
            The apparent magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.
        t_mission : u.year
            The duration of the mission, by default 2 years.
        stellar_variability : float
            The standard deviation of the variability of the star
            on timescales of ~1hr, assumed to be white noise, by default 10e-6.
        min_transits : int, optional
            The minimum number of transits required to detect planet, by default 2.
            If the number of transits is below this threshold, the efficiency is
            set to 0.
        mode : str, optional
            The mode to calculate the detection efficiency, either
            "linear" or "gamma", by default "linear".
        Returns
        -------
        float | np.ndarray
            The detection efficiency as a function of the SNR.

        Raises
        ------
        ValueError
            Raised if mode is not "linear" or "gamma".
        """
        if kwargs_detection_efficiency is None:
            kwargs_detection_efficiency = {}

        # calculate signal-to-noise ratio for N and N+1 transits
        snr_args = {
            "r_planet": r_planet,
            "r_star": r_star,
            "porb": porb,
            "m_star": m_star,
            "magnitude_v": magnitude_v,
            "n_cameras": n_cameras,
            "t_mission": t_mission,
            "stellar_variability": stellar_variability,
        }
        snr_n_transits = self.calculate_snr(**snr_args)
        snr_n_plus_one_transits = self.calculate_snr(**snr_args, extra_transit=True)
        snr_values = [snr_n_transits, snr_n_plus_one_transits]

        # Calculate number of certain transits (N) and probability of N+1 transits
        n_transits = self.calculate_number_of_transits(t_mission, porb)
        p_n_plus_one_transits = self.calculate_p_N_plus_1_transits(t_mission, porb)
        p_n_transits = 1 - p_n_plus_one_transits

        # Calculate detection efficiencies for N and N+1 transits, depending on
        # detection efficiency model
        detection_eff = []
        if mode == "linear":
            for snr in snr_values:
                detection_eff.append(
                    self.linear_detection_efficiency(
                        snr,
                        **kwargs_detection_efficiency,
                    )
                )
        elif mode == "gamma":
            for snr in snr_values:
                detection_eff.append(
                    self.gamma_detection_efficiency(
                        snr,
                        **kwargs_detection_efficiency,
                    )
                )
        else:
            raise ValueError("Invalid mode. Choose 'linear' or 'gamma'.")

        # Set detection efficiency to 0 if number of transits is below threshold
        detection_eff[0] = np.where(n_transits >= min_transits, detection_eff[0], 0)
        detection_eff[1] = np.where(n_transits + 1 >= min_transits, detection_eff[1], 0)

        # Return weighted average of detection efficiencies
        return (
            p_n_transits * detection_eff[0] + p_n_plus_one_transits * detection_eff[1]
        )
