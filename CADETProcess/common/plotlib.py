import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from CADETProcess import CADETProcessError

from .dataStructure import StructMeta
from .parameter import List, UnsignedFloat, String, Tuple

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

from matplotlib import cycler
chromapy_cycler = cycler(color=color_list)

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
        plt.rcParams['figure.figsize'] = (5,3)
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.prop_cycle'] = chromapy_cycler
    elif style == 'medium':
        plt.rcParams['figure.figsize'] = (10,6)
        plt.rcParams['lines.linewidth'] = 4
        plt.rcParams['font.size'] = 24
        plt.rcParams['axes.prop_cycle'] = chromapy_cycler
    elif style == 'large':
        plt.rcParams['figure.figsize'] = (15,9)
        plt.rcParams['lines.linewidth'] = 6
        plt.rcParams['font.size'] = 30
        plt.rcParams['axes.prop_cycle'] = chromapy_cycler
    else:
        raise CADETProcessError('Not a valid style')

def plot(x, y, plot_parameters=None, show=False, save_path=None):
    """Helper function to create plot.

    Parameters
    ----------
    x : np.array
        The x axis data.
    y : array
       The y axis data.
    plot_parameters : PlotParameters
       Information about plot
    """
    if plot_parameters is None:
        plot_parameters = PlotParameters()

    set_style(this.style)

    plt.figure()
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_parameters.fill_regions is not None:
        for fill in plot_parameters.fill_regions:
            color = color_list[fill['color_index']]
            plt.fill_between([fill['start'], fill['end']],
                             fill['y_max'], alpha=0.3, color=color)
            if fill['start'] < plot_parameters.xlim[0]:
                x_position = (plot_parameters.xlim[0] + fill['end'])/2
            else:
                x_position = (fill['start'] + fill['end'])/2
            y_position = 0.5 * fill['y_max']
            plt.text(x_position, y_position, fill['text'],
                     horizontalalignment='center',
                     verticalalignment='center')

    if plot_parameters.overlay is not None:
        for x_over, y_over in plot_parameters.overlay:
            plt.plot(x, y_over, linewidth=0.5, alpha=0.5)
            plt.gca().set_prop_cycle(None)

    if plot_parameters.hlines is not None:
        for line in plot_parameters.hlines:
            plt.hlines(line['y'], line['x_min'], line['x_max'])

    if plot_parameters.y_ticks is not None:
        locs = np.array([y_tick['loc'] for y_tick in plot_parameters.y_ticks])
        labels = [y_tick['label'] for y_tick in plot_parameters.y_ticks]
        plt.yticks(locs, labels)

    if plot_parameters.x_ticks is not None:
        locs = np.array([x_tick['loc'] for x_tick in plot_parameters.x_ticks])
        labels = [x_tick['label'] for x_tick in plot_parameters.x_ticks]
        plt.xticks(locs, labels, rotation=72, horizontalalignment='center')

    if plot_parameters.annotations is not None:
        for annotation in plot_parameters.annotations:
            ax.annotate(annotation['text'],
                        xy=annotation['xy'],
                        xycoords='data',
                        xytext=annotation['xytext'],
                        textcoords='offset points',
                        arrowprops=annotation['arrowstyle']
                        )


    plt.plot(x,y)

    plt.xlabel(plot_parameters.x_label)
    plt.ylabel(plot_parameters.y_label)
    plt.xlim(plot_parameters.xlim)
    plt.ylim(plot_parameters.ylim)
    plt.title(plot_parameters.title)

    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


class PlotParameters(metaclass=StructMeta):
    title = String()
    x_label = String()
    x_ticks = List()
    y_label = String()
    y_ticks = List()
    xlim = Tuple()
    ylim = Tuple()
    end = UnsignedFloat()
    fill_regions = List()
    overlay = List()
    hlines = List()
    annotations = List()
