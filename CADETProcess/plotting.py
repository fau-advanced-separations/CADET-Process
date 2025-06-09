"""
=======================================
Plotting (:mod:`CADETProcess.plotting`)
=======================================

.. currentmodule:: CADETProcess.plotting

This module provides functionality for plotting in CADET-Process.

General Style
=============

.. autosummary::
    :toctree: generated/

    set_figure_style
    SecondaryAxis
    Layout
    set_layout

Setup Figure
============

.. autosummary::
    :toctree: generated/

    setup_figure
    create_and_save_figure


Annotations
===========

.. autosummary::
    :toctree: generated/

    Annotation
    add_annotations

Ticks
=====

.. autosummary::
    :toctree: generated/

    Tick
    set_yticks
    set_xticks

Fill Regions
============

.. autosummary::
    :toctree: generated/

    FillRegion
    add_fill_regions

Text
====

.. autosummary::
    :toctree: generated/

    add_text

Hlines
======

.. autosummary::
    :toctree: generated/

    HLines
    add_hlines

"""  # noqa

import sys
from functools import wraps
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import (
    Callable,
    Integer,
    List,
    String,
    Structure,
    Tuple,
    UnsignedFloat,
)

this = sys.modules[__name__]

style = "medium"

color_dict = {
    "blue": matplotlib.colors.to_rgb("#000099"),
    "red": matplotlib.colors.to_rgb("#990000"),
    "green": matplotlib.colors.to_rgb("#009900"),
    "orange": matplotlib.colors.to_rgb("#D79B00"),
    "purple": matplotlib.colors.to_rgb("#896999"),
    "grey": matplotlib.colors.to_rgb("#444444"),
}
color_list = list(color_dict.values())
chromapy_cycler = cycler(color=color_list)

linestyle_cycler = cycler("linestyle", ["--", ":", "-."])

textbox_props = dict(facecolor="white", alpha=1)


figure_styles = {
    "small": {
        "width": 5,
        "height": 3,
        "linewidth": 2,
        "font_small": 8,
        "font_medium": 10,
        "font_large": 12,
        "color_cycler": chromapy_cycler,
    },
    "medium": {
        "width": 10,
        "height": 6,
        "linewidth": 4,
        "font_small": 20,
        "font_medium": 24,
        "font_large": 28,
        "color_cycler": chromapy_cycler,
    },
    "large": {
        "width": 15,
        "height": 9,
        "linewidth": 6,
        "font_small": 25,
        "font_medium": 30,
        "font_large": 40,
        "color_cycler": chromapy_cycler,
    },
}


def set_figure_style(style: Optional[str] = "medium") -> None:
    """
    Define the sytle of a plot.

    Can set the sytle of a plot for small, medium and large plots. The
    figuresize of the figure, the linewitdth and color of the lines and the
    size of the font can be changed by switching the style.

    Parameters
    ----------
    style : str, optional
        Style of a figure plot, default set to small.

    Raises
    ------
    CADETProcessError
        If no valid style has been chosen as parameter.
    """
    if style not in figure_styles:
        raise CADETProcessError("Not a valid style")

    width = figure_styles[style]["width"]
    height = figure_styles[style]["height"]
    linewidth = figure_styles[style]["linewidth"]
    font_small = figure_styles[style]["font_small"]
    font_medium = figure_styles[style]["font_medium"]
    font_large = figure_styles[style]["font_large"]
    color_cycler = figure_styles[style]["color_cycler"]

    plt.rcParams["figure.figsize"] = (width, height)
    plt.rcParams["lines.linewidth"] = linewidth
    plt.rcParams["font.size"] = font_small  # controls default text sizes
    plt.rcParams["axes.titlesize"] = font_small  # fontsize of the axes title
    plt.rcParams["axes.labelsize"] = font_medium  # fontsize of the x and y labels
    plt.rcParams["xtick.labelsize"] = font_small  # fontsize of the tick labels
    plt.rcParams["ytick.labelsize"] = font_small  # fontsize of the tick labels
    plt.rcParams["legend.fontsize"] = font_small  # legend fontsize
    plt.rcParams["figure.titlesize"] = font_large  # fontsize of the figure title
    plt.rcParams["axes.prop_cycle"] = color_cycler


set_figure_style()


def get_fig_size(
    n_rows: Optional[int] = 1,
    n_cols: Optional[int] = 1,
    style: Optional[str] = None,
) -> tuple[float, float]:
    """
    Get figure size for figures with multiple Axes.

    Parameters
    ----------
    n_rows : int, optional
        Number of rows in the figure. The default is 1.
    n_cols : int, optional
        Number of columns in the figure. The default is 1.
    style : str, optional
        Style to use for the figure. The default is None.

    Returns
    -------
    fig_size : tuple
        Size of the figure (width, height)
    """
    if style is None:
        style = this.style

    width = figure_styles[style]["width"]
    height = figure_styles[style]["height"]

    fig_size = (n_cols * width + 2, n_rows * height + 2)

    return fig_size


def setup_figure(
    n_rows: Optional[int] = 1,
    n_cols: Optional[int] = 1,
    style: Optional[str] = None,
    squeeze: Optional[bool] = True,
) -> tuple[Figure, Axes]:
    """
    Set up a matplotlib figure with specified dimensions and style.

    Parameters
    ----------
    n_rows : int, optional
        Number of rows in the figure, by default 1.
    n_cols : int, optional
        Number of columns in the figure, by default 1.
    style : str, optional
        Style to use for the figure. Uses a predefined style if None.
    squeeze : bool, optional
        If True, extra dimensions are removed from the returned Axes array, by default True.

    Returns
    -------
    tuple[Figure, Axes]
        A tuple containing the Figure object and an Axes object or an array of Axes objects.
    """
    if style is None:
        style = this.style
    set_figure_style(style)

    fig_size = get_fig_size(n_rows, n_cols)
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, squeeze=squeeze, figsize=fig_size
    )

    fig.tight_layout()

    return fig, axs


class SecondaryAxis(Structure):
    """Parameters for secondary axis."""

    components = List()
    y_label = String()
    y_lim = Tuple()
    transform = Callable()


class Layout(Structure):
    """General figure layout."""

    style = String()
    title = String()
    x_label = String()
    x_ticks = List()
    y_label = String()
    y_ticks = List()
    x_lim = Tuple()
    y_lim = Tuple()


def set_layout(
    ax: Axes,
    layout: Layout,
    show_legend: bool = True,
    ax_secondary: Optional[SecondaryAxis] = None,
    secondary_layout: Optional[Layout] = None,
) -> None:
    """
    Configure the layout of a matplotlib Axes object.

    Parameters
    ----------
    ax : Axes
        The primary matplotlib Axes object to configure.
    layout : Layout
        Layout object containing axis labels, limits, title, and ticks.
    show_legend : bool, optional
        Whether to display the legend. Default is True.
    ax_secondary : Optional[SecondaryAxis], optional
        The secondary Axes object, if applicable.
    secondary_layout : Optional[Layout], optional
        Layout object for the secondary axis, if applicable.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(layout.x_label)
    ax.set_ylabel(layout.y_label)
    ax.set_xlim(layout.x_lim)
    ax.set_ylim(layout.y_lim)
    ax.set_title(layout.title)

    if layout.x_ticks is not None:
        set_xticks(layout.x_ticks)
    if layout.y_ticks is not None:
        set_yticks(layout.y_ticks)

    lines, labels = ax.get_legend_handles_labels()

    if ax_secondary is not None:
        ax_secondary.set_ylabel(secondary_layout.y_label)
        ax_secondary.set_ylim(secondary_layout.y_lim)

        if show_legend:
            lines_secondary, labels_secondary = ax_secondary.get_legend_handles_labels()
            ax_secondary.legend(
                lines_secondary + lines, labels_secondary + labels, loc=0
            )
    else:
        if show_legend and len(labels) != 0:
            ax.legend()


class Tick(Structure):
    """Parameters for Axes ticks."""

    location: Tuple()
    label: String()


def set_yticks(ax: Axes, y_ticks: list[Tick]) -> None:
    """
    Set the y-ticks on a matplotlib Axes object.

    Parameters
    ----------
    ax : Axes
        The matplotlib Axes object to set the y-ticks on.
    y_ticks : list[Tick]
        List of Tick objects containing location and label for each y-tick.
    """
    locs = np.array([y_tick.location for y_tick in y_ticks])
    labels = [y_tick.label for y_tick in y_ticks]
    ax.set_yticks(locs, labels)


def set_xticks(ax: Axes, x_ticks: list[Tick]) -> None:
    """
    Set the x-ticks on a matplotlib Axes object with rotation.

    Parameters
    ----------
    ax : Axes
        The matplotlib Axes object to set the x-ticks on.
    x_ticks : list[Tick]
        List of Tick objects containing location and label for each x-tick.
    """
    locs = np.array([x_tick.location for x_tick in x_ticks])
    labels = [x_tick.label for x_tick in x_ticks]
    plt.xticks(locs, labels, rotation=72, horizontalalignment="center")


def add_text(
    ax: Axes,
    text: str,
    position: tuple[float, float] = (0.05, 0.9),
    tb_props: Optional[Any] = None,
    **kwargs: Optional[dict],
) -> None:
    """
    Add text to a matplotlib Axes object.

    Parameters
    ----------
    ax : Axes
        The matplotlib Axes object to add text to.
    text : str
        The text to be added.
    position : tuple[float], optional
        The position of the text, default is (0.05, 0.9).
    tb_props : Optional[Any], optional
        Properties to update the textbox with.
    **kwargs : Optional[dict]
        Additional keyword arguments for text customization.
    """
    if tb_props is not None:
        textbox_props.update(tb_props)

    ax.text(
        *position,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=textbox_props,
        **kwargs,
    )


def add_overlay(
    ax: Axes,
    y_overlay: npt.ArrayLike,
    x_overlay: Optional[npt.ArrayLike] = None,
    **plot_args: Optional[dict],
) -> None:
    """
    Add overlay plot(s) to a matplotlib Axes object.

    Parameters
    ----------
    ax : Axes
        The matplotlib Axes object to which the overlay is added.
    y_overlay : npt.ArrayLike
        The y-data for the overlay plot(s).
    x_overlay : Optional[list], optional
        The x-data for the overlay plot(s). If None, uses x-data from the first line in ax.
    **plot_args : Optional[dict]
        Additional keyword arguments for customizing the plot.
    """
    y_overlay = np.array(y_overlay)

    if x_overlay is None:
        x_overlay = ax.lines[0].get_xdata()

    for y_over in y_overlay:
        ax.plot(x_overlay, y_over, **plot_args)
        ax.set_prop_cycle(None)


class Annotation(Structure):
    """Parameters for text annotations."""

    text = String()
    xy = Tuple()
    xytext = Tuple()
    arrowstyle = "-|>"


def add_annotations(
    ax: Axes,
    annotations: list[Annotation],
) -> None:
    """Add list of annotations to axis ax."""
    for annotation in annotations:
        ax.annotate(
            annotation.text,
            xy=annotation.xy,
            xycoords="data",
            xytext=annotation.xytext,
            textcoords="offset points",
            arrowprops={
                "arrowstyle": annotation.arrowstyle,
            },
        )


class FillRegion(Structure):
    """Parameters for fill region."""

    color_index = Integer()
    start = UnsignedFloat()
    end = UnsignedFloat()

    y_max = UnsignedFloat()

    text = String()


def add_fill_regions(
    ax: Axes,
    fill_regions: list[FillRegion],
    x_lim: Optional[npt.ArrayLike] = None,
) -> None:
    """Add FillRegion to axes."""
    for fill in fill_regions:
        color = color_list[fill.color_index]
        ax.fill_between(
            [fill.start, fill.end],
            fill.y_max,
            alpha=0.3,
            color=color,
        )

        if fill.text is not None:
            if x_lim is None or fill.start < x_lim[0]:
                x_position = (x_lim[0] + fill.end) / 2
            else:
                x_position = (fill.start + fill.end) / 2
            y_position = 0.5 * fill.y_max

            ax.text(
                x_position,
                y_position,
                fill.text,
                horizontalalignment="center",
                verticalalignment="center",
            )


class HLines(Structure):
    """Parameters for plotting horizontal lines."""

    y = UnsignedFloat()
    x_min = UnsignedFloat()
    x_max = UnsignedFloat()


def add_hlines(ax: Axes, hlines: list[HLines]) -> None:
    """Add hlines to matplotlib Axes."""
    for line in hlines:
        ax.hlines(line.y, line.x_min, line.x_max)


def create_and_save_figure(func: Callable) -> Callable:
    """Wrap plot functions to provide some general utility."""

    @wraps(func)
    def wrapper(
        *args: Any,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        show: bool = True,
        file_name: Optional[str] = None,
        style: str = "medium",
        **kwargs: Any,
    ) -> tuple:
        """
        Wrap plot functions to provide some general utility.

        Parameters
        ----------
        *args :
            Parameters wrapped around.
        fig : Figure, optional
            Figure object.
        ax : Axes, optional
           Axes to plot on. If None, a new standard figure will be created.
        show : bool, optional
            If True, show plot. The default is False.
        file_name : str, optional
            Path for saving figure. If None, figure is not saved.
        style : str, optional
            Style for figure. Default i 'medium'.
        **kwargs :
            Additional parameters wrapped around.

        """
        if ax is None:
            fig, ax = setup_figure(style=style)

        func(*args, ax=ax, **kwargs)

        if fig is not None:
            fig.tight_layout()

        if file_name is not None:
            plt.savefig(file_name)

            plt.close(fig)
            if show:
                dummy = plt.figure(figsize=fig.get_size_inches())
                new_manager = dummy.canvas.manager
                new_manager.canvas.figure = fig
                fig.set_canvas(new_manager.canvas)
                plt.show()

        return fig, ax

    return wrapper
