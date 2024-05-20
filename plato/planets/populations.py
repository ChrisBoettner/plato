from typing import Any, Optional

import numpy as np
import pandas as pd
from methodtools import lru_cache
from scipy.stats import norm
from astropy import units as u

from plato.instrument.detection import DetectionModel


class PopulationModel:
    """
    A class to represent the NGPPS planet populations.
    """

    def __init__(
        self,
        stellar_population: pd.DataFrame,
        num_embryos: int,
        snapshot_age: int = int(1e8),
        detection_model: Optional[DetectionModel] = None,
        keep_all_columns: bool = False,
        keep_invalid: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the planet population model.

        Parameters
        ----------
        stellar_population : pd.DataFrame
            A dataframe containing the properties of the stellar
            population. Must contain  the columns
                - "R_star": the radius of the star, in solar radii.
                - "M_star": the mass of the star, in solar masses.
                - "[Fe/H]": the metallicity of the star.
                - "Magnitude_V": the apparent magnitude of the star.
                - "sigma_star": the standard deviation of the
                                variability of the star.
                - "u1": the first limb darkening coefficient.
                - "u2": the second limb darkening coefficient.
                - "n_cameras": the number of cameras observing the star.
            Additionally, the column "gaiaID_DR3" will be used if present.
        num_embryos : int
            The number of planetary embryos in the NGPPS run, used to
            select the appropriate NGPPS snapshot. Can be 10, 20, 50,
            or 100.
        snapshot_age : int, optional
            The age of the planetary system in the NGPPS run, used to select
            the appropriate NGPPS snapshot, by default 100Myr.
        detection_model : Optional[DetectionModel], optional
            A detection model to use for the mock observations, by default None.
            If None, detection model with default parameters is used.
        keep_all_columns : bool, optional
            If True, all columns in the stellar population DataFrame are
            kept. Otherwise, only the required columns are kept, by default False.
        keep_invalid : bool, optional
            If True, invalid planetary systems are kept in the NGPPS population.
            Otherwise, invalid systems are removed, by default False.
        **kwargs : Any
            Additional keyword arguments to pass to the get_system_metallicities
            method.

        """

        self.detection_model = detection_model if detection_model else DetectionModel()

        # create stellar population attribute
        stellar_population_columns = [
            "R_star",
            "M_star",
            "[Fe/H]",
            "Magnitude_V",
            "sigma_star",
            "u1",
            "u2",
            "n_cameras",
        ]

        if "gaiaID_DR3" in stellar_population.columns:
            stellar_population_columns += ["gaiaID_DR3"]

        for col in stellar_population_columns:
            if col not in stellar_population.columns:
                raise ValueError(
                    f"Stellar population DataFrame must contain the column '{col}'."
                )
        if not keep_all_columns:
            stellar_population = stellar_population[stellar_population_columns]

        self.stellar_population = stellar_population

        # create planetary system population attribute
        ngpps_mapping = {
            10: "ng96",
            20: "ng74",
            50: "ng075",
            100: "ng76",
        }

        ngpps_population = pd.read_csv(
            "/home/chris/Documents/Projects/plato/data/external/NGPPS/"
            f"{ngpps_mapping[num_embryos]}/"
            f"snapshot_{snapshot_age}.csv"
        )

        if not keep_invalid:
            ngpps_population = ngpps_population[ngpps_population["valid"]]

        if not keep_all_columns:
            ngpps_population = ngpps_population[
                [
                    "system_id",
                    "planet_id",
                    "total_radius",
                    "total_mass",
                    "semi_major_axis",
                ]
            ]

        ngpps_population = ngpps_population.rename(
            columns={
                "total_radius": "R_planet",
                "total_mass": "M_planet",
                "semi_major_axis": "a",
            }
        )
        ngpps_population["R_planet"] = (
            ngpps_population["R_planet"].to_numpy() * u.Rjup
        ).to(u.Rearth)

        self.ngpps_population = ngpps_population

        self.system_populations, self.system_num_planets = self.group_systems()

        # create planetary system metallicity attribute(s)
        self.system_metallicity = self.get_system_metallicities(**kwargs)

    def group_systems(
        self,
    ) -> tuple[list[pd.DataFrame], list[int]]:
        """
        Group the planetary systems in the NGPPS population by
        system_id. Returns a list of DataFrames, where each
        DataFrame contains the planets in a single planetary
        system. Also returns a list of the number of planets
        in each system.

        Returns
        -------
        tuple[list[pd.DataFrame], list[int]]
            A tuple containing the list of DataFrames, and the
            list of the number of planets in each system.
        """

        grouping = self.ngpps_population.groupby("system_id")
        groups = [grouping.get_group(name) for name in grouping.groups]
        len_groups = [len(group) for group in groups]

        return groups, len_groups

    def get_system_metallicities(
        self,
        solar_gas_to_dust: float = 0.0149,
        get_all_columns: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieve planetary system metallicities from the NGPPS
        metadata.

        Parameters
        ----------
        solar_gas_to_dust : float, optional
            The solar gas-to-dust ratio, used to convert the
            gas-to-dust ratio of the planetary system to a
            metallicity, by default 0.0149.
        get_all_columns : bool, optional
            Whether to return all columns in the NGPPS metadata
            DataFrame, or just the 'system_id' and '[Fe/H]'. By
            default False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the system_id and metallicity
            of each planetary system in the NGPPS. If get_all_columns
            is True, all metadata columns are returned.
        """

        # read planetary system metadata
        metadata = pd.read_csv(
            "/home/chris/Documents/Projects/plato/data/external/NGPPS/"
            "NGPPS_variables.txt",
            header=None,
            sep=r"\s+",
            names=[
                "system_id",
                "mstar",
                "sigma",
                "expo",
                "ain",
                "aout",
                "fpg",
                "mwind",
            ],
        )

        # modify the 'system_id' column to remove 'system_id' prefix
        metadata["system_id"] = metadata["system_id"].str[3:].astype(int)

        # convert columns to float
        for col in metadata.columns[1:]:
            metadata[col] = metadata[col].map(lambda x: float(x.split("=")[1]))

        # calculate the metallicity of the planetary systems
        metadata["[Fe/H]"] = np.log10(
            metadata["fpg"] / solar_gas_to_dust
        )  # NGPPS II paper Eq. 2

        if get_all_columns:
            return metadata
        else:
            return metadata[["system_id", "[Fe/H]"]]

    @lru_cache(maxsize=1)
    def calculate_probabilities(
        self,
        decay_parameter: float = 10,
        correct_for_initial_distribution: bool = True,
        return_cumulative_probabilities: bool = False,
    ) -> np.ndarray:
        """
        Calculate the probability system assigment probabilities for each
        star in the stellar population. The probability distribution are
        calculated for all stars in the stellar population, based on the
        metallicities of the star and NGPPS systems.

        Parameters
        ----------
        decay_parameter : float
            The rate parameter of the exponential distribution
            used to calculate the probabilities of assigning a
            system to a star. With a value of 10, the probability
            of assigning a system drops by a factor of 10 for each
            0.1 dex difference in metallicity.
        correct_for_initial_distribution : bool, optional
            Whether to correct the probabilities for the initial
            metallicity distribution of the NGPPS systems, which
            is given by a normal distribution with mean -0.02 and
            standard deviation 0.22 (NGPPS paper 2), by default True.
        return_cumulative_probabilities : bool, optional
            If True, the cumulative probabilities are returned, by
            default False.

        Returns
        -------
        np.ndarray
            A 2D array containing the probabilities of assigning
            any system to any star in the stellar population. If
            return_cumulative_probabilities is True, the cumulative
            probabilities are returned instead.

        """
        # Efficiently compute the probability matrix using broadcasting
        diff_matrix = (
            self.stellar_population["[Fe/H]"].to_numpy()[:, None]
            - self.system_metallicity["[Fe/H]"].to_numpy()[None, :]
        )
        probabilities = np.power(10, -float(decay_parameter) * np.abs(diff_matrix))
        if correct_for_initial_distribution:
            original_distribution = norm(loc=-0.02, scale=0.22)
            probabilities /= original_distribution.pdf(  # type: ignore
                self.system_metallicity["[Fe/H]"]
            )

        probabilities /= probabilities.sum(axis=1, keepdims=True)  # Normalize row-wise

        if return_cumulative_probabilities:
            return np.cumsum(probabilities, axis=1)
        return probabilities

    def assign_random_systems(
        self,
        decay_parameter: float = 10,
        correct_for_initial_distribution: bool = True,
    ) -> np.ndarray:
        """
        Assign a random planetary system to each star in the stellar
        population, based on the metallicity distribution of the
        NGPPS systems. The log probability of a system being assigned
        to a star is proportional to the absolute difference in
        metallicity between the system and the star.

        The `decay_parameter` controls the rate at which the
        probability of assigning a system to a star decreases with
        the metallicity difference. A higher value results in a faster
        decrease in probability with metallicity difference.

        Parameters
        ----------
        decay_parameter : float
            The rate parameter of the exponential distribution
            used to calculate the probabilities of assigning a
            system to a star, by default 10. With a value of 10,
            the probability of assigning a system drops by a factor
            of 10 for each 0.1 dex difference in metallicity.
        correct_for_initial_distribution : bool, optional
            Whether to correct the probabilities for the initial
            metallicity distribution of the NGPPS systems, which
            is given by a normal distribution with mean -0.02 and
            standard deviation 0.22 (NGPPS paper 2), by default True.
        Returns
        -------
        np.ndarray
            A random sample of system ids assigned to each star in
            the stellar population.
        """
        cumulative_probabilities = self.calculate_probabilities(
            decay_parameter=decay_parameter,
            correct_for_initial_distribution=correct_for_initial_distribution,
            return_cumulative_probabilities=True,
        )  # type: ignore

        # generate random values for each row
        random_values = np.random.rand(cumulative_probabilities.shape[0], 1)

        # Use broadcasting to compare random values with cumulative probabilities
        random_indices = (random_values < cumulative_probabilities).argmax(axis=1)

        # system ids are between 1 and 1000, so add 1 to the indices
        system_ids = random_indices + 1
        return system_ids

    def create_mock_population(
        self,
        decay_parameter: float = 10,
        metallicity_limit: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Create a mock population of planetary systems by assigning
        random planetary systems to stars in the stellar population.
        The assigment probability is [Fe/H]-dependent, and calculated
        in the calculate_probabilities method. The planetary systems
        are also assigned random inclination angles.

        The assigned planets are checked for Roche limit crossing,
        and any planets that cross the Roche limit are removed from
        the mock population.

        Parameters
        ----------
        decay_parameter : float
            The rate parameter of the exponential distribution
            used to calculate the probabilities of assigning a
            system to a star, by default 10. With a value of 10,
            the probability of assigning a system drops by a factor
            of 10 for each 0.1 dex difference in metallicity.
        metallicity_limit : Optional[float], optional
            The minimum metallicity of the star to be included in
            the mock population. If None, no limit is
            applied, by default None.

        Returns
        -------
        pd.DataFrame
            The mock population of planetary systems, with the assigned
            planetary systems, stellar properties, and random inclination
            angles.
        """

        # assign random systems to stars
        system_ids = self.assign_random_systems(decay_parameter=decay_parameter)

        # assign random inlcination angles (cos(i) is uniform between 0 and 1)
        random_cos_i = np.random.rand(self.stellar_population.shape[0])

        # get the systems corresponding to the assigned system ids
        # (random indices differ by 1 from system ids)
        systems = pd.concat([self.system_populations[ind - 1] for ind in system_ids])
        num_planets = [self.system_num_planets[ind - 1] for ind in system_ids]

        stellar_properties = self.stellar_population.copy()
        stellar_properties["cos_i"] = random_cos_i

        stellar_properties = stellar_properties.loc[
            np.repeat(
                stellar_properties.index,
                num_planets,
            )
        ].reset_index(drop=True)

        systems = pd.concat(
            [
                systems.reset_index(drop=True),
                stellar_properties.reset_index(drop=True),
            ],
            axis=1,
        )

        # remove impossible planets
        roche_limit = self.detection_model.transit_model.calculate_Roche_limit(
            m_p=systems["M_planet"].to_numpy() * u.Mearth,
            r_p=systems["R_planet"].to_numpy() * u.Rearth,
            m_star=systems["M_star"].to_numpy() * u.Msun,
            r_star=systems["R_star"].to_numpy() * u.Rsun,
        )
        systems = systems[systems["a"] > roche_limit]

        if metallicity_limit:
            systems = systems[systems["[Fe/H]"] > metallicity_limit]
        return systems.reset_index(drop=True)

    def create_mock_observation(
        self,
        decay_parameter: float = 10,
        metallicity_limit: Optional[float] = None,
        result_format: str = "minimal",
        remove_zeros: bool = True,
    ) -> pd.DataFrame:
        """
        Create a mock observation of the NGPPS population.
        First creates a mock population using the create_mock_population
        method, then calculates the detection efficiency of each planet
        in the mock population using the detection model.

        Parameters
        ----------
        decay_parameter : float
            The rate parameter of the exponential distribution
            used to calculate the probabilities of assigning a
            system to a star, by default 10. With a value of 10,
            the probability of assigning a system drops by a factor
            of 10 for each 0.1 dex difference in metallicity.
        metallicity_limit : Optional[float], optional
            The minimum metallicity of the star to be included in
            the mock population. If None, no limit is
            applied, by default None.
        result_format : str, optional
            The format of the result DataFrame. Must be one of
            'minimal', 'reconstructable', or 'full'.
            The available formats are:
                - 'minimal': contains the planet radius, mass,
                             semi-major axis, and detection efficiency.
                - 'reconstructable': contains the 'minimal' columns,
                                     plus the cos_i, system_id,
                                     planet_id, and gaiaID_DR3. Can be
                                     used to reconstruct all columns.
                - 'full': contains all columns in the mock population,
                          including the stellar parameters,
            by default "minimal".
        remove_zeros : bool, optional
            If True, planets with a detection efficiency of 0 are
            removed from the mock population, by default True.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the mock observation of the NGPPS
            population, at the specified level of detail.
        """

        mock_population = self.create_mock_population(
            decay_parameter=decay_parameter,
            metallicity_limit=metallicity_limit,
        )
        mock_population["detection_efficiency"] = (
            self.detection_model.detection_efficiency(mock_population)
        )

        if remove_zeros:
            mock_population = mock_population[
                mock_population["detection_efficiency"] > 0
            ]

        if result_format == "minimal":
            mock_population = mock_population[
                [
                    "R_planet",
                    "M_planet",
                    "a",
                    "detection_efficiency",
                ]
            ]
        elif result_format == "reconstructable":
            mock_population = mock_population[
                [
                    "R_planet",
                    "M_planet",
                    "a",
                    "detection_efficiency",
                    "cos_i",
                    "system_id",
                    "planet_id",
                    "gaiaID_DR3",
                ]
            ]
        elif result_format == "full":
            pass
        else:
            raise ValueError(
                "return_full must be one of 'minimal', 'reconstructable', or 'full'."
            )

        return mock_population.reset_index(drop=True)
