import pandas as pd
from typing import Optional, Callable


def quality_cuts(
    dataframe: pd.DataFrame,
    max_error: float = 0.2,
    error_type: str = "relative",
    remove_nans: bool = True,
    remove_negative_parallaxes: bool = True,
    checked_columns: list = [
        "ra",
        "dec",
        "pmra",
        "pmdec",
        "parallax",
        "radial_velocity",
    ],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply quality cuts to the input dataframe. The cuts are based
    on the relative or absolute errors of the columns in the
    checked_columns list. The stars with errors larger than
    max_error are removed. Error columns must be named as
    {column}_error.


    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.
    max_error : float, optional
        The maximum relative or absolute error allowed
        for the columns in checked_columns, by default 0.2.
    error_type : str, optional
        The error type, either 'relative' or 'absolute',
        by default 'relative'.
    remove_nans : bool, optional
        If True, remove rows with NaN values in the
        checked_columns, by default True.
    remove_negative_parallaxes : bool, optional
        If True, remove stars with negative parallaxes,
        by default True.
    checked_columns : list, optional
        The columns to apply the quality cuts to, by default
        ["ra", "dec", "pmra", "pmdec", "parallax", "radial_velocity"].
    verbose : bool, optional
        If True, print the number of stars removed, by default True.

    Returns
    -------
    pd.DataFrame
        The input dataframe with the quality cuts applied.

    """
    new_df = dataframe.copy()

    # remove NaN values
    if remove_nans:
        new_df = new_df.dropna(subset=checked_columns)

    # remove negative and zero parallaxes
    if remove_negative_parallaxes:
        new_df = new_df[new_df["parallax"] >= 0]

    # remove stars with large errors
    for column in checked_columns:
        if error_type == "relative":
            error = new_df[f"{column}_error"] / new_df[column]
        elif error_type == "absolute":
            error = new_df[f"{column}_error"]
        else:
            raise ValueError(
                f"Error type {error_type!r} not recognized."
                " Must be 'relative' or 'absolute'."
            )
        if column in new_df.columns:
            new_df = new_df[error < max_error]
        else:
            raise ValueError(f"Column {column!r} not found in dataframe.")

    if verbose:
        print(
            f"Removed {len(dataframe) - len(new_df)}/{len(dataframe)} stars based on "
            f"quality cuts ({(len(dataframe) - len(new_df)) / len(new_df) * 100:.1f}%)."
        )

    return new_df


def filter_valid_targets(
    target_dataframe: pd.DataFrame,
    conditions: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Filter the target dataframe to only include
    valid entries.


    Parameters
    ----------
    target_dataframe : pd.DataFrame
        The target dataframe to filter.
    conditions : Optional[Callable], optional
        A function that takes a dataframe and
        returns a boolean mask of valid targets,
        by default None. If None, the default
        conditions are used, which are:
            - Stellar Type is FGK
            - Radius > 0
            - Mass > 0
            - Teff > 0
            - logg is not null
            - [Fe/H] is not null

    Returns
    -------
    pd.DataFrame
        The target dataframe filtered for valid entries.
    """

    if conditions is None:
        conditions = (
            lambda dataframe: (dataframe["Stellar Type"] == "FGK")
            & (dataframe["Radius"] > 0)
            & (dataframe["Mass"] > 0)
            & (dataframe["Teff"] > 0)
            & dataframe["logg"].notnull()
            & dataframe["[Fe/H]"].notnull()
        )

    return target_dataframe[conditions(target_dataframe)]


def update_field_dataframe(
    all_sky_dataframe: pd.DataFrame,
    field: str,
) -> pd.DataFrame:
    """
    Convinience function to update the field target dataframes.
    Given the all sky dataframe and the field name, load
    the field target dataframe and return the intersection
    entries from the all sky dataframe that are in the field
    target dataframe (based on the gaiaID_DR3 column).

    Parameters
    ----------
    all_sky_dataframe : pd.DataFrame
        The all sky dataframe.
    field : str
        The field name, should be LOPS2 or LOPN1,

    Returns
    -------
    pd.DataFrame
        The all sky dataframe with only the stars in the field
        target dataframe.
    """

    field_dataframe = pd.read_csv(
        f"/home/chris/Documents/Projects/plato/data/processed/{field}_targets.csv"
    )

    update_field_dataframe = all_sky_dataframe[
        all_sky_dataframe["gaiaID_DR3"].isin(field_dataframe["gaiaID_DR3"])
    ]

    # add n_cameras column back in
    update_field_dataframe["n_cameras"] = field_dataframe["n_cameras"]
    return update_field_dataframe
