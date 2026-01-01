"""
=======================================
Plotting (:mod:`CADETProcess.plotting`)
=======================================

.. currentmodule:: CADETProcess.plotting

This module provides functionality for plotting in CADET-Process.

General Utils
=============

.. autosummary::
    :toctree: generated/

    get_fig_size
    setup_figure
    get_all_twin_handles_labels
    show_or_reopen
    style_and_save_figure

Secondary Axis
==============

.. autosummary::
    :toctree: generated/

    SecondaryAxis

Text
====

.. autosummary::
    :toctree: generated/

    add_text

Annotations
===========

.. autosummary::
    :toctree: generated/

    annotate


Fill Regions
============

.. autosummary::
    :toctree: generated/

    fill_between

"""  # noqa

import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Literal, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib import cycler

# %% Colors

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

# %% Fig size


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


# %% Layout

# Figure sizes from Elsevier style guide:
# https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing
figure_layouts = {
    "minimal": {
        "width": mm_to_inches(30),
        "height": mm_to_inches(20),
        "linewidth": 1.0,          # Axes, spines, grid
        "marker_size": 3,          # For line plots
        "font_small": 8,           # Ticks, legend
        "font_medium": 10,         # Axis labels
        "font_large": 12,          # Title
        "tick_length": 3,          # Tick mark length
        "color_cycler": chromapy_cycler,
    },
    "1_col": {
        "width": mm_to_inches(90),
        "height": mm_to_inches(60),
        "linewidth": 1.0,
        "marker_size": 5,
        "font_small": 8,
        "font_medium": 10,
        "font_large": 12,
        "tick_length": 4,
        "color_cycler": chromapy_cycler,
    },
    "1.5_col": {
        "width": mm_to_inches(140),
        "height": mm_to_inches(93.33),
        "linewidth": 1.2,
        "marker_size": 7,
        "font_small": 9,
        "font_medium": 11,
        "font_large": 14,
        "tick_length": 5,
        "color_cycler": chromapy_cycler,
    },
    "2_col": {
        "width": mm_to_inches(190),
        "height": mm_to_inches(126.67),
        "linewidth": 1.5,
        "marker_size": 9,
        "font_small": 10,
        "font_medium": 12,
        "font_large": 16,
        "tick_length": 6,
        "color_cycler": chromapy_cycler,
    },
}


# %% Figure size

def get_fig_size(
    layout: Literal["1_col", "1_5_col", "2_col"] = "1_col",
    nrows: int = 1,
    ncols: int = 1,
    aspect: float | None = None,
    scale_with_subplots: bool = False,
    padding: float = 0.0,
    figsize: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Compute a publication-ready figure size in inches.

    Parameters
    ----------
    layout: Literal["1_col", "1_5_col", "2_col"] = "1_col",
        Figure layout.
    nrows : int
        Number of subplot rows.
    ncols : int
        Number of subplot columns.
    aspect : float | None
        Width / height ratio. If provided, overrides preset height.
    scale_with_subplots : bool
        If True, multiply width/height by ncols/nrows.
    padding : float
        Extra inches to add.
    figsize : tuple[float, float] | None
        Override width/height directly.

    Returns
    -------
    width, height : tuple[float, float]
    """
    if figsize is not None:
        width, height = figsize
    else:
        if layout not in figure_layouts:
            raise ValueError(
                f"Invalid layout {layout}. Options: {list(figure_layouts)}"
            )
        width = figure_layouts[layout]["width"]
        height = figure_layouts[layout]["height"]

    if aspect is not None:
        height = width / aspect

    if scale_with_subplots:
        width = ncols * width + padding
        height = nrows * height + padding

    return width, height


# %% Style

@contextmanager
def mpl_style_context(
    layout: Literal["1_col", "1_5_col", "2_col"] = "1_col",
) -> None:
    """Context manager for temporary matplotlib rc parameters."""
    layout_settings = figure_layouts[layout]
    rc_params = {
        "axes.titlesize": layout_settings["font_large"],
        "axes.labelsize": layout_settings["font_medium"],
        "axes.prop_cycle": layout_settings["color_cycler"],
        "figure.titlesize": layout_settings["font_large"],
        "font.size": layout_settings["font_small"],
        "legend.fontsize": layout_settings["font_small"],
        "lines.linewidth": layout_settings["linewidth"],
        "lines.markersize": layout_settings["marker_size"],
        "xtick.labelsize": layout_settings["font_small"],
        "ytick.labelsize": layout_settings["font_small"],
    }
    with mpl.rc_context(rc_params):
        yield


def setup_figure(
    layout: Literal["1_col", "1_5_col", "2_col"] = "1_col",
    nrows: int = 1,
    ncols: int = 1,
    aspect: float | None = None,
    scale_with_subplots: bool = False,
    padding: float = 0.0,
    figsize: tuple[float, float] | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes | npt.NDArray[plt.Axes]]:
    """
    Set up a matplotlib figure with local styling and flexible options.

    Parameters
    ----------
    layout : Literal["1_col", "1_5_col", "2_col"] = "1_col",
        Figure layout.
    nrows : int
        Number of subplot rows.
    ncols : int
        Number of subplot columns.
    aspect : float | None
        Width / height ratio. If provided, overrides preset height.
    scale_with_subplots: bool = False
        If True, scale figure size by ncols/nrows.
    padding : float
        Extra inches to add.
    figsize : tuple[float, float] | None
        Override width/height directly.
    *args
        Additional arguments for `plt.subplots`.
    **kwargs
        Additional keyword arguments for `plt.subplots`.

    Returns
    -------
    tuple[plt.Figure, plt.Axes | npt.NDArray[plt.Axes]]
        Figure and Axes object(s).
    """
    figsize = get_fig_size(
        layout,
        nrows,
        ncols,
        aspect,
        scale_with_subplots,
        padding,
        figsize,
    )
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        *args,
        **kwargs,
    )
    return fig, ax


# %% Twin Axis

def get_twins(
    ax: plt.Axes,
    axis: Literal["x", "y", "both"] = "x",
) -> list[plt.Axes]:
    """
    Get twin axes of a specified axis.

    Parameters
    ----------
    ax : plt.Axes
        Axes to get twins for.
    axis : Literal["x", "y"], default="x"
        Which twins to get .

    Returns
    -------
    `list`[plt.Axes]
    """
    axs = []
    match axis:
        case "x":
            axs = ax.get_shared_x_axes().get_siblings(ax)
        case "y":
            axs = ax.get_shared_y_axes().get_siblings(ax)
    return list(reversed(
        [a for a in axs if (a is not ax) & (a.bbox.bounds == ax.bbox.bounds)]
    ))


def get_all_twin_handles_labels(ax: plt.Axes) -> tuple[list, list]:
    """
    Return handles and labels from an axes and all of its twins.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The reference axes.

    Returns
    -------
    handles : list
        All line/patch artists from `ax` and its twin axes.
    labels : list of str
        Corresponding legend labels.

    """
    # Matplotlib groups axes that share x or y; twins belong to these groups.
    sec_axs = get_twins(ax)

    axs = [ax, *sec_axs]
    handles = []
    labels = []

    for a in axs:
        h, l = a.get_legend_handles_labels()  # noqa: E741
        handles.extend(h)
        labels.extend(l)

    return handles, labels


def offset_secondary_yaxes(
    ax: plt.Axes,
    spacing_factor: float = 0.05,
    min_spacing: float = 0.2,
    max_spacing: float = 0.5,
    side: Literal["left", "right"] = "right",
) -> None:
    """
    Offset secondary y-axis spines to avoid overlap, scaling with axes size.

    Parameters
    ----------
    ax : plt.Axes
        The primary axes.
    side : {"right", "left"}, optional
        Side on which to place the secondary spines. Default is "right".
    spacing_factor : float, optional
        Factor to scale the spacing with the axis width. Default is 0.05.
    max_spacing : float, optional
        Maximum spacing. Default is 0.2.
    side : {"right", "left"}, optional
        Side on which to place the secondary spines. Default is "right".

    Raises
    ------
    ValueError
        If `side` is not "right" or "left".
    """
    fig = ax.get_figure()
    twin_axes = get_twins(ax)

    # Get axis dimensions in figure coordinates
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_width = bbox.width

    # Calculate dynamic spacing
    sign = 1 if side == "right" else -1
    base = 1.0 if side == "right" else 0.0

    spacing = min(max(spacing_factor * ax_width, min_spacing), max_spacing)

    for i, sec_ax in enumerate(twin_axes[1:], start=1):
        spine = sec_ax.spines[side]
        position = base + sign * i * spacing

        if sec_ax.yaxis.get_label_text():
            position += 0.02

        spine.set_position(("axes", position))


# %% Figure Utils

def show_or_reopen(fig: plt.Figure) -> None:
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


def figure_utils(func: Callable) -> Callable:
    """
    Unified decorator for styling, and saving figures.

    Returns
    -------
    Callable
        Decorator function.
    """
    @wraps(func)
    def figure_utils_wrapper(
        *args: Any,
        ax: Optional[plt.Axes | npt.NDArray[plt.Axes]] = None,
        setup_figure_kwargs: Optional[dict] = None,
        file_name: Optional[os.PathLike] = None,
        dpi: int = 300,
        show: bool = True,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes | npt.NDArray[plt.Axes]]:
        """
        Apply styles, save, and optionally create a figure.

        Parameters
        ----------
        *args : Any
            Additional positional arguments passed to the wrapped function.
        ax : Optional[plt.Axes], default=None
            Optional Matplotlib Axes.
            If not provided, a new figure is created.
        setup_figure_kwargs : Optional[dict], default=None
            Additional options to setup the figure.
        file_name : Optional[os.PathLike], default=None
            File name to store the figure. If None is provided, the figure is
            not saved.
        dpi : int, default=300
            DPI for saving the figure.
        show : bool, default=True
            If False, close the figure.
        tight_layout : bool, default=True
            If True, set tight layout.
        **kwargs : Any
            Additional keyword arguments passed to the wrapped function.

        Returns
        -------
        tuple[plt.Figure, plt.Axes | npt.NDArray[plt.Axes]]
            The Matplotlib Figure and Axes objects.
        """
        setup_figure_kwargs = {
            "layout": "1_col",
            "scale_with_subplots": True,
            **(setup_figure_kwargs or {}),
        }

        with mpl_style_context(setup_figure_kwargs["layout"]):
            fig, ax = func(
                *args,
                ax=ax,
                setup_figure_kwargs=setup_figure_kwargs,
                **kwargs
            )
            if tight_layout:
                fig.tight_layout()

        if file_name is not None:
            fig.savefig(file_name, dpi=dpi)

        if show:
            fig.show()
        else:
            plt.close(fig)

        return fig, ax
    return figure_utils_wrapper


# %% Secondary Axis

@dataclass
class SecondaryAxis:
    """Convenience class for secondary axis configuration."""

    components: list[str]
    ylabel: str | None = None
    ylim: tuple[float, float] | None = None
    transform: Callable | None = None


# %% Text

textbox_props = dict(facecolor="white", alpha=1)


def add_text(
    ax: plt.Axes,
    text: str,
    position: tuple[float, float] = (0.05, 0.9),
    *,
    tb_props: dict | None = None,
    **kwargs: Any,
) -> None:
    """
    Add text to a matplotlib Axes object.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib Axes object to add text to.
    text : str
        The text to be added.
    position : tuple[float, float], default=(0.05, 0.9)
        The position of the text in axes coordinates.
    tb_props : dict | None, default=None
        Dictionary of properties to update the textbox (e.g., `boxstyle`, `facecolor`).
    **kwargs
        Additional keyword arguments for `ax.text`.
    """
    tb_props = {**textbox_props, **(tb_props or {})}

    ax.text(
        *position,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=tb_props,
        **kwargs,
    )


# %% Annotations

def annotate(
    ax: plt.Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    *,
    arrowstyle: str = "-|>",
    **kwargs: Any,
) -> None:
    """
    Add an annotation to a matplotlib Axes object.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib Axes object to annotate.
    text : str
        The annotation text.
    xy : tuple[float, float]
        The point (x, y) to annotate.
    xytext : tuple[float, float]
        The position (x, y) of the annotation text.
    arrowstyle : str, default="-|>"
        Style of the arrow connecting `xy` and `xytext`.
    **kwargs
        Additional keyword arguments for `ax.annotate`.
    """
    ax.annotate(
        text,
        xy=xy,
        xycoords="data",
        xytext=xytext,
        textcoords="offset points",
        arrowprops={"arrowstyle": arrowstyle},
        **kwargs,
    )


# %% Fill regions

def fill_between(
    ax: plt.Axes,
    start: float,
    end: float,
    y_max: float,
    alpha: float = 0.3,
    *,
    color_index: int | None = None,
    text: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Add fill region with optional text labeling.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on.
    start : float
        Start x-position of the fill region.
    end : float
        End x-position of the fill region.
    y_max : float
        Maximum y-value for the fill.
    alpha : float, default=0.3
        Transparency of the fill.
    color_index : int | None, default=None
        Index into a predefined color list. If None, uses `color` from `kwargs`.
    text : str | None, default=None
        Optional label for the region.
    **kwargs
        Additional arguments passed to `ax.fill_between`.
    """
    color = color_list[color_index] if color_index is not None else kwargs.pop("color", None)

    ax.fill_between(
        [start, end],
        y_max,
        alpha=alpha,  # Use the `alpha` parameter, not hardcoded 0.3
        color=color,
        **kwargs,
    )

    if text is not None:
        x_position = (start + end) / 2
        x_lim = ax.get_xlim()
        if start < x_lim[0]:
            x_position = (x_lim[0] + end) / 2
        elif end > x_lim[1]:
            x_position = (start + x_lim[1]) / 2

        y_position = 0.5 * y_max

        ax.text(
            x_position,
            y_position,
            text,
            ha="center",
            va="center",
        )
