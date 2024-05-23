import pandas as pd
from typing import Optional


class PlanetPopulationMetrics:
    """
    Class to calculate planet population metrics.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize the PlanetPopulationMetrics class.

        """

    def get_subpopulation(
        self,
        planet_population: pd.DataFrame,
        population: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter the data by population. Column
        "Population" must be present in the data.

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        population : Optional[str], optional
            Name of the population to filter by,
            by default None
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.DataFrame
            Subset of the data filtered by population.
        """
        if population:
            return planet_population[planet_population["Population"] == population]
        else:
            return planet_population

    def calculate_number_of_planets(
        self,
        planet_population: pd.DataFrame,
        population: Optional[str] = None,
        remove_dwarfs: bool = True,
    ) -> pd.Series:
        """
        Calculate the number of planets for each category.

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        number_of_systems : int
            The number of systems for which the population was
            determined.
        population : Optional[str], optional
            Name of the population to filter by,
            by default None
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.Series
            Series containing the number of planets for each category.
        """
        planet_pop = self.get_subpopulation(
            planet_population,
            population,
        )
        num_planets = planet_pop["Planet Category"].value_counts()

        if remove_dwarfs and "Dwarf" in num_planets.index:
            num_planets = num_planets.drop(index="Dwarf")

        return num_planets

    def calculate_fraction_of_systems_with_planet(
        self,
        planet_population: pd.DataFrame,
        number_of_systems: int,
        population: Optional[str] = None,
        remove_dwarfs: bool = True,
    ) -> pd.Series:
        """
        Calculate the fraction of systems with at least one planet
        (per category).

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        number_of_systems : int
            The number of systems for which the population was
        population : Optional[str], optional
            Name of the population to filter by,
            by default None
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.Series
            Series containing the fraction of systems with at least one
            planet for each category.
        """
        planet_pop = self.get_subpopulation(
            planet_population,
            population,
        )
        systems_with_planet_type = planet_pop.groupby("Planet Category")[
            "TargetID"
        ].nunique()
        fraction_systems_with_planet = systems_with_planet_type / number_of_systems

        if remove_dwarfs and "Dwarf" in fraction_systems_with_planet.index:
            fraction_systems_with_planet = fraction_systems_with_planet.drop(
                index="Dwarf"
            )

        return fraction_systems_with_planet

    def calculate_occurrence_rate(
        self,
        planet_population: pd.DataFrame,
        number_of_systems: int,
        population: Optional[str] = None,
        remove_dwarfs: bool = True,
    ) -> pd.Series:
        """
        Calculate the occurrence rate (average number of planets per system)
        for each category.

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        number_of_systems : int
            The number of systems for which the population was
            determined.
        population : Optional[str], optional
            Name of the population to filter by,
            by default None
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.Series
            Series containing the occurrence rate for each category.
        """

        number_of_planets = self.calculate_number_of_planets(
            planet_population,
            population,
            remove_dwarfs,
        )
        occurrence_rate = number_of_planets / number_of_systems

        if remove_dwarfs and "Dwarf" in occurrence_rate.index:
            occurrence_rate = occurrence_rate.drop(index="Dwarf")

        return occurrence_rate

    def calculate_multiplicity(
        self,
        planet_population: pd.DataFrame,
        population: Optional[str] = None,
        remove_dwarfs: bool = True,
    ) -> pd.Series:
        """
        Calculate the multiplicity (average number of planets per system
        with at least one planet) for each category.

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        population : Optional[str], optional
            Name of the population to filter by,
            by default None
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.Series
            Series containing the multiplicity for each category.
        """

        planet_pop = self.get_subpopulation(
            planet_population,
            population,
        )
        multiplicity = planet_pop.groupby("Planet Category").apply(
            lambda x: len(x) / x["TargetID"].nunique()
        )
        if remove_dwarfs and "Dwarf" in multiplicity.index:
            multiplicity = multiplicity.drop(index="Dwarf")
        return multiplicity

    def calculate_metrics(
        self,
        planet_population: pd.DataFrame,
        number_of_systems: int,
        population: Optional[str] = None,
        remove_dwarfs: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate all metrics for the planet population, meaning the
            - number of planets
            - fraction of systems with planet
            - occurrence rate
            - multiplicity

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        number_of_systems : int
            The number of systems for which the population was
            determined.
        population : Optional[str], optional
            Name of the population to filter by,
            by default None
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the calculated metrics.
        """

        num_planets = self.calculate_number_of_planets(
            planet_population,
            population,
            remove_dwarfs,
        )
        fraction_systems_with_planet = self.calculate_fraction_of_systems_with_planet(
            planet_population,
            number_of_systems,
            population,
            remove_dwarfs,
        )
        occurrence_rate = self.calculate_occurrence_rate(
            planet_population,
            number_of_systems,
            population,
            remove_dwarfs,
        )
        multiplicity = self.calculate_multiplicity(
            planet_population,
            population,
            remove_dwarfs,
        )

        metrics = pd.DataFrame(
            {
                "Number of Planets": num_planets,
                "System Fraction": fraction_systems_with_planet,
                "Occurrence Rate": occurrence_rate,
                "Multiplicity": multiplicity,
            }
        )

        return metrics

    def calculate_agg_metrics(
        self,
        planet_populations: list[pd.DataFrame],
        number_of_systems: int,
        population: Optional[str] = None,
        remove_dwarfs: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate the statistics of the metrics for a list
        of planet populations. Specifically, calculate the
        median, 16th, and 84th percentiles of the metrics.

        Parameters
        ----------
        planet_populations : list[pd.DataFrame]
            List of planet populations, each containing columns
            "TargetID", "Planet Category", and optionally "Population",
            if filtering by population is desired.
        number_of_systems : int
            The number of systems for which the population was
            determined.
        population : Optional[str], optional
            Name of the population to filter by,
            by default None
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the statistics of the metrics.

        """
        results: list | pd.DataFrame = []
        for planet_population in planet_populations:
            results.append(
                self.calculate_metrics(
                    planet_population,
                    number_of_systems,
                    population,
                    remove_dwarfs,
                )
            )

        results = pd.concat(results)

        quantiles = [0.5, 0.16, 0.84]
        agg_results = results.groupby("Planet Category").agg(
            [lambda x: x.quantile(q) for q in quantiles]
        )

        agg_results.rename(
            {
                "<lambda_0>": "Median",
                "<lambda_1>": "16th",
                "<lambda_2>": "84th",
            },
            inplace=True,
        )
        return agg_results
