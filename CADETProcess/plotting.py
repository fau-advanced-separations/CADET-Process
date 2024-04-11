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

"""

from functools import wraps
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cycler

from CADETProcess import CADETProcessError

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    Integer, List, String, Tuple, Callable, UnsignedFloat
)


this = sys.modules[__name__]

style = 'medium'

color_dict = {
    'blue': matplotlib.colors.to_rgb('#000099'),
    'red':  matplotlib.colors.to_rgb('#990000'),
    'green': matplotlib.colors.to_rgb('#009900'),
    'orange': matplotlib.colors.to_rgb('#D79B00'),
    'purple': matplotlib.colors.to_rgb('#896999'),
    'grey':  matplotlib.colors.to_rgb('#444444'),
         }
color_list = list(color_dict.values())
chromapy_cycler = cycler(color=color_list)

linestyle_cycler = cycler('linestyle', ['--', ':', '-.'])

textbox_props = dict(facecolor='white', alpha=1)


figure_styles = {
    'small': {
        'width': 5,
        'height': 3,
        'linewidth': 2,
        'font_small': 8,
        'font_medium': 10,
        'font_large': 12,
        'color_cycler': chromapy_cycler,

    },
    'medium': {
        'width': 10,
        'height': 6,
        'linewidth': 4,
        'font_small': 20,
        'font_medium': 24,
        'font_large': 28,
        'color_cycler': chromapy_cycler,
    },
    'large': {
        'width': 15,
        'height': 9,
        'linewidth': 6,
        'font_small': 25,
        'font_medium': 30,
        'font_large': 40,
        'color_cycler': chromapy_cycler,
    },
}


def set_figure_style(style='medium'):
    """Define the sytle of a plot.

    Can set the sytle of a plot for small, medium and large plots. The
    figuresize of the figure, the linewitdth and color of the lines and the
    size of the font can be changed by switching the style.

    Parameters
    ----------
    style : str
        Style of a figure plot, default set to small.

    Raises
    ------
    CADETProcessError
        If no valid style has been chosen as parameter.
    """
    if style not in figure_styles:
        raise CADETProcessError('Not a valid style')

    width = figure_styles[style]['width']
    height = figure_styles[style]['height']
    linewidth = figure_styles[style]['linewidth']
    font_small = figure_styles[style]['font_small']
    font_medium = figure_styles[style]['font_medium']
    font_large = figure_styles[style]['font_large']
    color_cycler = figure_styles[style]['color_cycler']

    plt.rcParams['figure.figsize'] = (width, height)
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['font.size'] = font_small          # controls default text sizes
    plt.rcParams['axes.titlesize'] = font_small     # fontsize of the axes title
    plt.rcParams['axes.labelsize'] = font_medium    # fontsize of the x and y labels
    plt.rcParams['xtick.labelsize'] = font_small    # fontsize of the tick labels
    plt.rcParams['ytick.labelsize'] = font_small    # fontsize of the tick labels
    plt.rcParams['legend.fontsize'] = font_small    # legend fontsize
    plt.rcParams['figure.titlesize'] = font_large   # fontsize of the figure title
    plt.rcParams['axes.prop_cycle'] = color_cycler


set_figure_style()


def get_fig_size(n_rows=1, n_cols=1, style=None):
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

    width = figure_styles[style]['width']
    height = figure_styles[style]['height']

    fig_size = (n_cols * width + 2, n_rows * height + 2)

    return fig_size


def setup_figure(n_rows=1, n_cols=1, style=None, squeeze=True):
    """
    Setup a figure.

    Parameters
    ----------
    n_rows : int, optional
        Number of rows in the figure. The default is 1.
    n_cols : int, optional
        Number of columns in the figure. The default is 1.
    style : str, optional
        Style to use for the figure. The default is None.
    squeeze : bool, optional
        If True, extra dimensions are squeezed out from the returned array of Axes.
        The default is True.

    Returns
    -------
    fig : Figure
    ax : Axes or array of Axes
    """
    if style is None:
        style = this.style
    set_figure_style(style)

    fig_size = get_fig_size(n_rows, n_cols)
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=squeeze,
        figsize=fig_size
    )

    fig.tight_layout()

    return fig, axs


class SecondaryAxis(Structure):
    components = List()
    y_label = String()
    y_lim = Tuple()
    transform = Callable()


class Layout(Structure):
    style = String()
    title = String()
    x_label = String()
    x_ticks = List()
    y_label = String()
    y_ticks = List()
    x_lim = Tuple()
    y_lim = Tuple()


def set_layout(
        ax,
        layout,
        show_legend=True,
        ax_secondary=None,
        secondary_layout=None):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
            lines_secondary, labels_secondary = \
                ax_secondary.get_legend_handles_labels()
            ax_secondary.legend(
                lines_secondary + lines, labels_secondary + labels, loc=0
            )
    else:
        if show_legend and len(labels) != 0:
            ax.legend()


class Tick(Structure):
    location: Tuple()
    label: String()


def set_yticks(ax, y_ticks):
    locs = np.array([y_tick['loc'] for y_tick in y_ticks])
    labels = [y_tick['label'] for y_tick in y_ticks]
    ax.set_yticks(locs, labels)


def set_xticks(ax, x_ticks):
    locs = np.array([x_tick['loc'] for x_tick in x_ticks])
    labels = [x_tick['label'] for x_tick in x_ticks]
    plt.xticks(locs, labels, rotation=72, horizontalalignment='center')


def add_text(ax, text, position=(0.05, 0.9), tb_props=None, **kwargs):
    if tb_props is not None:
        textbox_props.update(tb_props)

    ax.text(
        *position, text, transform=ax.transAxes, verticalalignment='top',
        bbox=textbox_props, **kwargs
    )


def add_overlay(ax, y_overlay, x_overlay=None, **plot_args):
    if not isinstance(y_overlay, list):
        y_overlay = [y_overlay]

    if x_overlay is None:
        x_overlay = ax.lines[0].get_xdata()

    for y_over in y_overlay:
        ax.plot(x_overlay, y_over, **plot_args)
        ax.set_prop_cycle(None)


class Annotation(Structure):
    text = String()
    xy = Tuple()
    xytext = Tuple()
    arrowstyle = '-|>'


def add_annotations(ax, annotations):
    for annotation in annotations:
        ax.annotate(
            annotation.text,
            xy=annotation.xy,
            xycoords='data',
            xytext=annotation.xytext,
            textcoords='offset points',
            arrowprops={
                'arrowstyle': annotation.arrowstyle
            }
        )


class FillRegion(Structure):
    color_index = Integer()
    start = UnsignedFloat()
    end = UnsignedFloat()

    y_max = UnsignedFloat()

    text = String()


def add_fill_regions(ax, fill_regions, x_lim=None):
    for fill in fill_regions:
        color = color_list[fill.color_index]
        ax.fill_between(
            [fill.start, fill.end],
            fill.y_max, alpha=0.3, color=color
        )

        if fill.text is not None:
            if x_lim is None or fill.start < x_lim[0]:
                x_position = (x_lim[0] + fill.end)/2
            else:
                x_position = (fill.start + fill.end)/2
            y_position = 0.5 * fill.y_max

            ax.text(
                x_position, y_position, fill.text,
                horizontalalignment='center',
                verticalalignment='center'
            )


class HLines(Structure):
    y = UnsignedFloat()
    x_min = UnsignedFloat()
    x_max = UnsignedFloat()


def add_hlines(ax, hlines):
    for line in hlines:
        ax.hlines(line.y, line.x_min, line.x_max)


def create_and_save_figure(func):
    @wraps(func)
    def wrapper(
            *args,
            fig=None,
            ax=None,
            show=True, file_name=None, style='medium',
            **kwargs):
        """Wrapper around plot function.

        Parameters
        ----------
        fig : Figure, optional
        ax : Axes, optional
           Axes to plot on. If None, a new standard figure will be created.
        show : bool, optional
            If True, show plot. The default is False.
        file_name : str, optional
            Path for saving figure. If None, figure is not saved.
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
