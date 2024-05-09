import os
from typing import Any, Optional

import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure


def set_plot_defaults() -> None:
    """
    Set plot defaults.

    """
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="pastel",
        font_scale=5,
        rc={
            "figure.figsize": (18.5, 10.5),
            "axes.grid": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[varg]{txfonts}",
        },
    )


def get_palette(
    n_colors: int = 6,
    start: float = 0,
    rot: float = 0.4,
    diverging: bool = False,
    as_cmap: bool = False,
    second_palette_start: float = 1.65,
    **kwargs: Any,
) -> list | ListedColormap:
    """
    Create cubehelix color palette using seaborn.

    Parameters
    ----------
    n_colors : int, optional
        Number of colors in the palette. The default is 6.
    start : float, optional
        The hue value at the start of the helix. The default is -.2.
    rot : float, optional
        Rotations around the hue wheel over the range of the palette. The default is
        0.4 .
    diverging : bool, optional
        If True, create a diverging colormap. The default is False.
    as_cmap: bool, optional
        If True, colormap is returned as matplotlib ListedColormap object. Otherwise
        its a seaborn ColorPalette. The default is False.
    second_palette_start : float, optional
        Starting point for second part of colormap, if diverging is True. The default
        is 1.65 .
    **kwargs : Any
        Additional parameters for cubehelix_palette. Ignored if diverging is True.

    Returns
    -------
    list | ListedColormap
        Return colormap either as seaborn color palette (list) or matplotlib colormap.

    """
    if not diverging:
        palette = sns.cubehelix_palette(
            n_colors=n_colors,
            start=start,
            rot=rot,
            as_cmap=as_cmap,  # type: ignore
            **kwargs,
        )
        return palette

    # for diverging colormap, create two maps and combine them
    else:
        # if output is ListedColormap, use max number of colors
        if as_cmap:
            n_colors = 256

        # create palettes
        palette_one = sns.cubehelix_palette(
            n_colors=(
                n_colors // 2 if n_colors % 2 else n_colors // 2 + 1
            ),  # odd number handling
            start=second_palette_start,
            rot=rot,
            light=0.95,
            as_cmap=False,
            reverse=True,
        )
        palette_two = sns.cubehelix_palette(
            n_colors=n_colors // 2,
            start=start,
            rot=rot,
            light=0.95,
            as_cmap=False,
        )
        diverging_palette = palette_one + palette_two

        if as_cmap:
            return ListedColormap(diverging_palette)
        return diverging_palette


def adjust_legend(ax: Axes, ncols: int = 3, pad: float = 1, **kwargs: Any) -> Axes:
    """
    Adjust plot to accomodate for legend. Can increase the number of columns for the
    legend, and add extra space at top of legend for legend.

    Parameters
    ----------
    ax : Axes
        The matplotlib Axes object.
    ncols : int, optional
        Number of columns for the legend. The default is 3.
    pad : float, optional
        Additional padding at top of plot (multiple of ymax). The default is 1,
        i.e. no change.
    kwargs : Any
        Additional parameter passed to ax.legend()

    Returns
    -------
    Axes
        Matplotlib Axes object with adjusted for the legend.

    """

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * pad)
    ax.legend(ncols=ncols, **kwargs)
    return ax


class FigureProcessor:
    """
    Class to handle figures created by seaborn

    """

    def __init__(self, figure: Figure | Axes, process: bool = True) -> None:
        """
        Load in and (optionally) process seaborn figure.

        Parameters
        ----------
        seaborn_plot : Figure
            The matplotlib figure object.
        process : bool, optional
            If True, calls process function which further processes the plot. The
            default is True.

        """
        if isinstance(figure, Axes):
            self.figure = figure.figure
        elif isinstance(figure, Figure):
            self.figure = figure
        else:
            raise TypeError("Input must be a matplotlib Figure or Axes object.")
        assert isinstance(self.figure, Figure)

        if process:
            self.process()

    def process(self) -> None:
        """
        Process image.

        """
        # align y labels
        assert isinstance(self.figure, Figure)
        self.figure.align_ylabels()

    def save(
        self,
        file_name: str,
        figure_directory: str,
        save: Optional[bool] = True,
    ) -> None:
        """
        Save figure in figures directory.

        Parameters
        ----------
        file_name : str
            Name and relative path of file in figures directory.
        figure_directory : str
            Path to figures directory.
        save: Optional[bool], optional
            If False, only create file structure and don't actually save image. The
            default is True.

        """
        path = os.path.join(figure_directory, file_name)

        # create directory if it doesn't exist already
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if save:
            assert isinstance(self.figure, Figure)
            self.figure.savefig(path, bbox_inches="tight", pad_inches=0)
