from multiprocessing import Pool
from typing import Any, Optional
from functools import partial

import polars
from astropy.coordinates import SkyCoord
from pandas import DataFrame as pdDataFrame
from polars import DataFrame as plDataFrame
from tqdm import tqdm

from plato.platopoint import platopoint  # type: ignore


def count_CCDs(source: SkyCoord, **kwargs: Any) -> int:
    """Count the number of Plato CCDs that a particular
    source falls on (for a given plato pointing which
    can be changed with kwargs). Default pointing is the
    LOPS2 field.


    Parameters
    ----------
    source : SkyCoord
        Coordinates of the source.
    kwargs : Any
        Additional arguments to pass to platopoint, including
        Plato pointing.

    Returns
    -------
    int
        Number of CCDs the source falls on.
    """
    # find CCD that the source falls on
    on_CCD = platopoint(targetCoord=source, **kwargs)[0].values()  # type: ignore
    # count number of CCDs
    if any(on_CCD):
        num_CCDs = sum(1 for val in on_CCD if val is not None)
    else:
        num_CCDs = 0
    return num_CCDs


def find_targets(
    data: pdDataFrame | plDataFrame,
    progress: bool = True,
    processes: Optional[int] = None,
    **kwargs: Any
) -> pdDataFrame | plDataFrame:
    """Find the number of Plato CCDs that each source falls on, and add
    this information to the DataFrame as "num_CCDs" column.

    Parameters
    ----------
    data : pandas.DataFrame | polars.DataFrame
        DataFrame containing the sources with 'ra' and 'dec' columns in deg.
    progress : bool, optional
        Show progress bar, by default True.
    processes : int, optional
        Number of processes, by default None (uses all available cores).
    kwargs : Any
        Additional arguments to pass to platopoint, including Plato pointing.

    Returns
    -------
    pdDataFrame | plDataFrame
        DataFrame with 'num_CCDs' column added. Zero corresponds to target
        being outside the Plato field of view.
    """

    if not isinstance(data, (pdDataFrame, plDataFrame)):
        raise ValueError("data must be a (pandas or polars) DataFrame.")

    try:
        data_coords = SkyCoord(ra=data["ra"], dec=data["dec"], unit="deg")
    except KeyError | polars.exceptions.ColumnNotFoundError:
        raise ValueError("data must be a DataFrame with 'ra' and 'dec' columns in deg.")

    targets = []

    with Pool(processes) as pool:
        targets = list(
            tqdm(
                pool.imap(partial(count_CCDs, **kwargs), data_coords),  # type: ignore
                total=len(data_coords),
                disable=not progress,
                desc="Targets: ",
            )
        )

    if isinstance(data, pdDataFrame):
        data["num_CCDs"] = targets
    else:
        data = data.with_columns(num_CCDs=polars.Series(targets))
    return data
