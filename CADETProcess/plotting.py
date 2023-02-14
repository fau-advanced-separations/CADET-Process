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

    set_style
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

from CADETProcess.dataStructure import StructMeta, Structure
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


def set_style(style='medium'):
    """Defines the sytle of a plot.

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
    if style == 'small':
        plt.rcParams['figure.figsize'] = (5, 3)
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.prop_cycle'] = chromapy_cycler
    elif style == 'medium':
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['lines.linewidth'] = 4
        plt.rcParams['font.size'] = 24
        plt.rcParams['axes.prop_cycle'] = chromapy_cycler
    elif style == 'large':
        plt.rcParams['figure.figsize'] = (15, 9)
        plt.rcParams['lines.linewidth'] = 6
        plt.rcParams['font.size'] = 30
        plt.rcParams['axes.prop_cycle'] = chromapy_cycler
    else:
        raise CADETProcessError('Not a valid style')


set_style()


def setup_figure(n_rows=1, n_cols=1, style=None):
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)

    if style is None:
        style = this.style
    set_style(style)

    fig.tight_layout()

    return fig, ax


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


class HLines(metaclass=StructMeta):
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

        ax = func(*args, ax=ax, **kwargs)

        if file_name is not None:
            plt.savefig(file_name)

            fig.tight_layout()

            plt.close(fig)
            if show:
                dummy = plt.figure(figsize=fig.get_size_inches())
                new_manager = dummy.canvas.manager
                new_manager.canvas.figure = fig
                fig.set_canvas(new_manager.canvas)
                plt.show()

        return fig, ax

    return wrapper
