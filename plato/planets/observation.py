from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas._typing import AggFuncTypeBase
from tqdm import tqdm

from plato.planets.populations import PopulationModel


class ObservationModel:
    """
    Class to create and analyze mock observations of exoplanet populations.

    """

    def __init__(
        self,
        load_mocks: bool = True,
        num_mocks: int = 300,
        mocks: Optional[dict] = None,
        metallicity_cut_mocks: Optional[dict] = None,
    ) -> None:
        """
        Initialize the ObservationModel class.

        Parameters
        ----------
        load_mocks : bool, optional
            If True, load mocks from disk, by default True.
        num_mocks : int, optional
            The number of mocks to load, by default 300.
        mocks : Optional[dict], optional
            Dictionary of mocks, to load. Dict keys are the number of embryos,
            while the values are lists of DataFrames. Overrides the loaded
            mocks, by default None.
        metallicity_cut_mocks : Optional[dict], optional
            Dictionary of mocks with metallicity cuts, to load. Dict keys are
            the number of embryos, while the values are lists of DataFrames.
            Overrides the loaded metallicity cut mocks, by default None.

        """
        self.mocks = None
        self.metallicity_cut_mocks = None

        if load_mocks:
            self.mocks = self.load_mocks(
                False,
                num_mocks=num_mocks,
            )
            self.metallicity_cut_mocks = self.load_mocks(
                True,
                num_mocks=num_mocks,
            )

        self.mocks = mocks if mocks is not None else self.mocks
        self.metallicity_cut_mocks = (
            metallicity_cut_mocks
            if metallicity_cut_mocks is not None
            else self.metallicity_cut_mocks
        )

    def create_mocks(
        self,
        targets: pd.DataFrame,
        num_embryos: int | list[int],
        num_mocks: int = 300,
        metallicity_limit: Optional[float] = None,
        progress: bool = True,
        save: bool = True,
        path: Optional[str] = None,
    ) -> dict[int, pd.DataFrame]:
        """
        Create mock observations of exoplanet populations.

        Parameters
        ----------
        targets : pd.DataFrame
            Dataframe of target stars.
        num_embryos : int | list[int]
            Number of embryos assumed in the planet
            population model. If a list is provided, mocks
            will be created for each value.
        num_mocks : int, optional
            Number of mocks to create, by default 300
        metallicity_limit : Optional[float], optional
            Metallicity limit for the mock observations, by default None.
        progress : bool, optional
            If True, show progress bar, by default True.
        save : bool, optional
            If True, save the mocks to disk, by default True.
        path : Optional[str], optional
            Path to save the mocks to, by default None, which
            saves to the default path.

        Returns
        -------
        dict[int, pd.DataFrame]
            Dictionary of mocks, where the keys are the number of embryos
            and the values are lists of DataFrames.
        """

        # set path
        if path is None:
            if metallicity_limit is None:
                path = "../data/interim/mock_observations"
            else:
                path = "../data/interim/mock_observations_halo_Fe_cut"

        num_embryos = [num_embryos] if isinstance(num_embryos, int) else num_embryos

        # create mocks
        mock_dict = {}
        for n in num_embryos:
            population_model = PopulationModel(
                targets,
                num_embryos=n,
                additional_columns=["Population"],
            )

            mocks = []
            for i in tqdm(
                range(num_mocks),
                desc=f"Creating mock observations for {n} embryos:",
                disable=not progress,
            ):
                mock = population_model.create_mock_observation(
                    add_planet_category=True,
                    additional_columns=["Population"],
                    metallicity_limit=metallicity_limit,
                )

                if save:
                    mock.to_csv(
                        f"{path}/{n}/{i}.csv",
                        index=False,
                    )
                mocks.append(mock)
            mock_dict[n] = mocks
        return mock_dict

    def load_mocks(
        self,
        select_metallicity_cut_mocks: bool,
        num_mocks: int = 300,
        num_embryos: int | list[int] = [10, 20, 50, 100],
    ) -> dict[int, list[pd.DataFrame]]:
        """
        Load mock observations of exoplanet populations from disk.

        Parameters
        ----------
        select_metallicity_cut_mocks : bool
            If True, load mocks with metallicity cuts, by default False.
        num_mocks : int, optional
            Number of mocks to load, by default 300.
        num_embryos : int | list[int], optional
            Number of embryos assumed in the planet
            population model. If a list is provided, mocks
            will be loaded for each value, by default [10, 20, 50, 100].

        Returns
        -------
        dict[int, list[pd.DataFrame]]
            Dictionary of mocks, where the keys are the number of embryos
            and the values are lists of DataFrames.
        """
        if select_metallicity_cut_mocks:
            path = (
                "/home/chris/Documents/Projects/plato/"
                "data/interim/mock_observations_halo_Fe_cut"
            )
        else:
            path = (
                "/home/chris/Documents/Projects/plato/" "data/interim/mock_observations"
            )

        num_embryos = [num_embryos] if isinstance(num_embryos, int) else num_embryos

        mocks = {}
        for n in num_embryos:
            mocks[n] = [pd.read_csv(f"{path}/{n}/{i}.csv") for i in range(num_mocks)]

        return mocks

    def bin_planets(
        self,
        num_embryos: int,
        period_bins: list[float] = [0.5, 1, 2, 5, 10, 20, 50, 100, 250, 500, 750],
        select_metallicity_cut_mocks: bool = False,
        population: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Bin planets by period and planet category.

        Parameters
        ----------
        num_embryos : int
            Number of embryos for the mock observations, loads the corresponding mocks
            from the ObservationModel.
        period_bins : list[float], optional
            List of period bins to bin the planets by, by default
            [0.5, 1, 2, 5, 10, 20, 50, 100, 250, 500, 750] (days).
        population : Optional[str], optional
            Sub-population to do the binning for, by default None, which bins all
            planets.

        Returns
        -------
        pd.DataFrame
            DataFrame of binned planets, and the number of planets in each bin.
        """

        mocks = (
            self.metallicity_cut_mocks if select_metallicity_cut_mocks else self.mocks
        )
        if mocks is None:
            raise ValueError("No mocks loaded. Please load mocks first.")

        binned_mocks = []
        for mock in mocks[num_embryos]:
            mock = mock.copy()
            if population is not None:
                mock = mock[mock["Population"] == population]
            mock["Porb_bin"] = pd.cut(
                mock["Porb"], bins=period_bins, labels=period_bins[1:]
            )
            binned_mocks.append(
                pd.DataFrame(
                    mock.groupby(
                        ["Planet Category", "Porb_bin"],
                        observed=False,
                    ).size(),
                    columns=["Count"],
                ).fillna(0)
            )

        return pd.concat(binned_mocks)

    def aggregrate_statistics(
        self,
        num_embryos: int | list[int],
        statistic: Optional[list[AggFuncTypeBase]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Aggregate statistics for the binned planets.

        Parameters
        ----------
        num_embryos : int | list[int]
            Number of embryos for the mock observations, loads the corresponding mocks
            from the ObservationModel. If a list is provided, the statistics are
            averaged over all the mocks.
        statistic : Optional[list[AggFuncTypeBase]], optional
            List of statistics to aggregate, by default ["mean", "std"].
        kwargs : Any
            Additional keyword arguments to pass to bin_planets
            (e.g. period_bins, population).

        Returns
        -------
        pd.DataFrame
            DataFrame of aggregated statistics for the binned planets. The
            dataframe has a multi-index with the planet category and period
            bin as the index, and the statistics as the columns.
        """

        if statistic is None:
            statistic = ["mean", "std"]
        statistic = [statistic] if isinstance(statistic, str) else statistic

        num_embryos = [num_embryos] if isinstance(num_embryos, int) else num_embryos

        dfs = []
        for num in num_embryos:
            binned_mocks = self.bin_planets(
                num_embryos=num,
                **kwargs,
            )

            stats_df = (
                binned_mocks.groupby(
                    ["Planet Category", "Porb_bin"],
                    observed=False,
                )
                .agg(statistic)
                .droplevel(0, axis=1)
            )
            dfs.append(pd.DataFrame(stats_df))

        # calculate mean
        result_df = dfs[0]
        for df in dfs[1:]:
            result_df = result_df.add(df, fill_value=0)
        result_df /= len(dfs)
        return pd.DataFrame(np.ceil(result_df)).astype(int).fillna(0)

    def stats_pivot_table(
        self,
        num_embryos: int | list[int],
        statistic: str,
        reindex: list = [
            "Earth",
            "Super-Earth",
            "Neptunian",
            "Sub-Giant",
            "Giant",
        ],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Create a pivot table of the aggregated statistics, for easy visualization.

        Parameters
        ----------
        num_embryos : int
            The number of embryos for the mock observations, loads the corresponding
            mocks from the ObservationModel. If a list is provided, the statistics
            are averaged over all the mocks.
        statistic : str
            The statistic to pivot the table on. Must be a string description of
            the statistic (e.g. "mean", "std").
        reindex : list, optional
            List of planet categories to reindex the pivot table with, by default
            ["Earth", "Super-Earth", "Neptunian", "Sub-Giant", "Giant"]. This also
            fills in any missing planet categories.

        Returns
        -------
        pd.DataFrame
            Pivot table of the aggregated statistics, with the period bins as the index,
            the planet category as the columns, and the statistic as the values.
        """

        stats_df = self.aggregrate_statistics(
            num_embryos=num_embryos,
            statistic=[statistic],
            **kwargs,
        )

        pivot_table = stats_df.pivot_table(
            index="Porb_bin",
            columns="Planet Category",
            values=statistic,
            observed=False,
            fill_value=0,
        ).T
        return pivot_table.reindex(reindex).fillna(0)

    def formatted_statistics(
        self,
        num_embryos: int | list[int],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Format the aggregated statistics for easy visualization, and
        annotation in plots.
        The formatted statistics are the mean and standard deviation
        of the binned planets. The mean and standard deviation are
        formatted as strings. If multiple values of num_embryos are
        provided, the formatted statistics are concatenated with a
        newline character.

        Parameters
        ----------
        num_embryos : int | list[int]
            Number of embryos for the mock observations, loads the corresponding
            mocks from the ObservationModel. If a list is provided, the formatted
            statistics are concatenated with a newline character.

        Returns
        -------
        pd.DataFrame
            Dataframe of formatted annotations for the aggregated statistics.
        """

        num_embryos = [num_embryos] if isinstance(num_embryos, int) else num_embryos

        annots = []
        for num in num_embryos:
            mean_pivot = (
                self.stats_pivot_table(num, "mean", **kwargs).astype(int).astype(str)
            )
            std_pivot = (
                self.stats_pivot_table(num, "std", **kwargs).astype(int).astype(str)
            )
            annots.append(mean_pivot + r" $\pm$ " + std_pivot)
        annoted_stats = annots[0]
        for df in annots[1:]:
            annoted_stats = annoted_stats + "\n" + df
        return annoted_stats
