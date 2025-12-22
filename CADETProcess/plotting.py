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
import os
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Any, Literal, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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


# %% Style

style = "single_column"

color_dict = {
    "blue": mpl.colors.to_rgb("#000099"),
    "red": mpl.colors.to_rgb("#990000"),
    "green": mpl.colors.to_rgb("#009900"),
    "orange": mpl.colors.to_rgb("#D79B00"),
    "purple": mpl.colors.to_rgb("#896999"),
    "grey": mpl.colors.to_rgb("#444444"),
}
color_list = list(color_dict.values())
chromapy_cycler = cycler(color=color_list)

linestyle_cycler = cycler("linestyle", ["--", ":", "-."])

textbox_props = dict(facecolor="white", alpha=1)


def mm_to_inches(mm: float) -> float:
    """
    Convert mm to inches.

    Parameters
    ----------
    mm : float
        Value in mm.

    Returns
    -------
    float
        Value in inches.
    """
    return mm / 25.4


figure_styles = {
    "single_column": {
        "width": mm_to_inches(90),
        "height": mm_to_inches(60),
        "linewidth": 1.0,
        "font_small": 8,    # Ticks, legend
        "font_medium": 10,  # Axis labels
        "font_large": 12,   # Title
        "color_cycler": chromapy_cycler,
    },
    "1.5_column": {
        "width": mm_to_inches(140),
        "height": mm_to_inches(93.33),
        "linewidth": 1.2,
        "font_small": 9,    # Ticks, legend
        "font_medium": 11,  # Axis labels
        "font_large": 14,   # Title
        "color_cycler": chromapy_cycler,
    },
    "double_column": {
        "width": mm_to_inches(190),
        "height": mm_to_inches(126.67),
        "linewidth": 1.5,
        "font_small": 10,   # Ticks, legend
        "font_medium": 12,  # Axis labels
        "font_large": 16,   # Title
        "color_cycler": chromapy_cycler,
    },
}


def get_fig_size(
    n_rows: int = 1,
    n_cols: int = 1,
    style: Optional[Literal["single_column", "1.5_column", "double_column"]] = "single_column",
    scale_with_subplots: Optional[bool] = False,
) -> tuple[float, float]:
    """
    Get figure size for figures with multiple Axes.

    Parameters
    ----------
    n_rows : int, optional
        Number of rows in the figure. The default is 1.
    n_cols : int, optional
        Number of columns in the figure. The default is 1.
    style : Optional[Literal["single_column", "1.5_column", "double_column"]]
        Figure style ("single_column", "1.5_column", or "double_column").
        The default is "single_column".
    scale_with_subplots: Optional[bool] = False
        If True, scale figure size with number of column / rows.

    Returns
    -------
    fig_size : tuple
        Size of the figure (width, height)
    """
    width = figure_styles[style]["width"]
    height = figure_styles[style]["height"]
    if scale_with_subplots:
        return (n_cols * width + 2, n_rows * height + 2)
    else:
        return (width, height)


def setup_figure(
    n_rows: Optional[int] = 1,
    n_cols: Optional[int] = 1,
    style: Optional[Literal["single_column", "1.5_column", "double_column"]] = "single_column",
    scale_with_subplots: Optional[bool] = False,
    squeeze: Optional[bool] = True,
    **kwargs: Any
) -> tuple[Figure, Axes]:
    """
    Set up a matplotlib figure with local styling and flexible options.

    Parameters
    ----------
    n_rows : int, optional
        Number of rows in the subplot grid.
    n_cols : int, optional
        Number of columns in the subplot grid.
    style : Literal["single_column", "1.5_column", "double_column"] = "single_column"
        Figure style ("single_column", "1.5_column", or "double_column").
    scale_with_subplots: Optional[bool] = False
        If True, scale figure size with number of column / rows.
    squeeze : bool, optional
        If True, extra dimensions are removed from the Axes array.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes object(s).
    """
    # Resolve figure dimensions
    fig_size = get_fig_size(n_rows, n_cols, style, scale_with_subplots)

    # Create figure
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=squeeze,
        figsize=fig_size,
        **kwargs,
    )

    fig.tight_layout()

    return fig, axs


@contextmanager
def mpl_style_context(
    style: Literal["single_column", "1.5_column", "double_column"] = "single_column",
) -> None:
    """Context manager to temporarily set matplotlib rc parameters for a given style."""
    style_settings = figure_styles[style]
    rc_params = {
        "figure.titlesize": style_settings["font_large"],
        "font.size": style_settings["font_small"],
        "axes.titlesize": style_settings["font_large"],
        "axes.labelsize": style_settings["font_medium"],
        "xtick.labelsize": style_settings["font_small"],
        "ytick.labelsize": style_settings["font_small"],
        "legend.fontsize": style_settings["font_small"],
        "axes.prop_cycle": style_settings["color_cycler"],
        "lines.linewidth": style_settings["linewidth"],
    }
    with mpl.rc_context(rc_params):
        yield


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


class SecondaryAxis(Structure):
    """Parameters for secondary axis."""

    components = List()
    y_label = String()
    y_lim = Tuple()
    transform = Callable()


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


# %% Ticks

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


# %% Text

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


# %% Overlay

def add_overlay(
    ax: Axes,
    y_overlay: npt.ArrayLike | list[npt.ArrayLike],
    x_overlay: Optional[npt.ArrayLike] = None,
    **plot_args: Optional[dict],
) -> None:
    """
    Add overlay plot(s) to a matplotlib Axes object.

    Parameters
    ----------
    ax : Axes
        The matplotlib Axes object to which the overlay is added.
    y_overlay : npt.ArrayLike | list[npt.ArrayLike]
        The y-data for the overlay plot(s).
    x_overlay : Optional[list], optional
        The x-data for the overlay plot(s). If None, uses x-data from the first line in ax.
    **plot_args : Optional[dict]
        Additional keyword arguments for customizing the plot.
    """
    if not isinstance(y_overlay, list):
        y_overlay = [y_overlay]

    if x_overlay is None:
        x_overlay = ax.lines[0].get_xdata()

    for y_over in y_overlay:
        ax.plot(x_overlay, y_over, **plot_args)
        ax.set_prop_cycle(None)


# %% Annotation

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


# %% FillRegion

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


# %% HLines
class HLines(Structure):
    """Parameters for plotting horizontal lines."""

    y = UnsignedFloat()
    x_min = UnsignedFloat()
    x_max = UnsignedFloat()


def add_hlines(ax: Axes, hlines: list[HLines]) -> None:
    """Add hlines to matplotlib Axes."""
    for line in hlines:
        ax.hlines(line.y, line.x_min, line.x_max)


def show_or_reopen(fig: Figure) -> None:
    """Show figure, reopening it in a GUI window if necessary."""
    if fig.number not in plt.get_fignums():
        dummy = plt.figure(figsize=fig.get_size_inches())
        manager = dummy.canvas.manager
        manager.canvas.figure = fig
        fig.set_canvas(manager.canvas)
        fig.show()
        plt.close(dummy)
    else:
        fig.show()


# %% Create and save figure decorator

def create_and_save_figure(func: Callable) -> Callable:
    """Wrap plot functions to provide some general utility."""

    @wraps(func)
    def wrapper(
        *args: Any,
        fig: Optional[Figure] = None,
        ax: Optional[Axes | npt.NDArray[Axes]] = None,
        setup_figure_kwargs: Optional[dict] = None,
        show: bool = True,
        file_name: Optional[os.PathLike] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | npt.NDArray[Axes]]:
        """
        Wrap plot functions to provide some general utility.

        Parameters
        ----------
        *args : Any
            Additional parameters to be passed to plot method.
        fig : Optional[Figure] = None
            Figure object.
        ax : Optional[Axes | npt.NDArray[Axes]] = None
            Axes to plot on. If None, a new axes will be created.
        setup_figure_kwargs : Optional[dict]
            Additional keyword arguments to pass to `setup_figure`.
        show : bool
            If True, show plot. The default is True.
        file_name : Optional[os.PathLike]
            Path for saving figure. If None, figure is not saved.
        **kwargs : Any
            Additional keyword parameters to be passed to plot method.

        Returns
        -------
        tuple[Figure, Axes | npt.NDArray[Axes]]
            The figure and axes objects used for plotting.
        """
        # Use context manager to set default styles locally
        with mpl_style_context(style):
            if ax is None:
                fig, ax = setup_figure(**setup_figure_kwargs)

            func(*args, ax=ax, **kwargs)

        if fig is not None:
            fig.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, dpi=300)

        if show:
            show_or_reopen(fig)
        else:
            plt.close(fig)

        return fig, ax

    return wrapper
