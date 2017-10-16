from __future__ import division
import sys
import logging
import matplotlib.pyplot as plt
import plotly
import plotly.offline as py
from ipywidgets import widgets
from IPython.display import display
from lens.plotting import (plot_distribution,
                           plot_cdf,
                           plot_pairdensity_mpl,
                           plot_correlation)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Check whether we are in a notebook environment
# this is a false positive if we are in the Jupyter console
IN_NOTEBOOK = 'ipykernel' in sys.modules

PADDING = '10px'
PLOT_HEIGHT = 400
PLOT_WIDTH = 600
DPI = 72


def render_plotly_js(fig, width=800, height=600):
    """Return the plotly html for a plot"""
    if isinstance(fig, plt.Axes):
        fig = fig.figure
    else:
        fig = fig

    if isinstance(fig, plt.Figure):
        fig = plotly.tools.mpl_to_plotly(fig, strip_style=True, resize=True)

    fig.layout['width'] = width
    fig.layout['height'] = height

    return py.plot(fig, output_type='div', include_plotlyjs=False,
                   show_link=False)


def create_correlation_plot_widget(ls):
    """Return a widget with correlation plot.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.

    Returns
    -------
    :class:`ipywidgets.Widget`
        Jupyter widget to explore correlation matrix plot.
    """
    fig = plot_correlation(ls)
    return widgets.HTML(render_plotly_js(fig, width=fig.layout['width'],
                                         height=fig.layout['height']),
                        height='{:.0f}px'.format(fig.layout['height']))


def update_plot(f, args, plot_area, **kwargs):
    """Updates the content of an output widget with rendered function"""

    fig = f(*args)
    fig.set_size_inches(PLOT_WIDTH / DPI, PLOT_HEIGHT / DPI)
    plot_area.clear_output()

    if 'height' in kwargs.keys():
        plot_area.layout.height = '{:.0f}px'.format(kwargs['height'])
    if 'width' in kwargs.keys():
        plot_area.layout.width = '{:.0f}px'.format(kwargs['width'])

    with plot_area:
        display(fig)


def _update_pairdensity_plot(ls, dd1, dd2, plot_area):
    if dd1.value != dd2.value:
        update_plot(plot_pairdensity_mpl,
                    [ls, dd1.value, dd2.value],
                    plot_area, height=600, width=600)


def create_pairdensity_plot_widget(ls):
    """Create a pairwise density widget.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.

    Returns
    -------
    :class:`ipywidgets.Widget`
        Jupyter widget to explore pairdensity plots.
    """
    numeric_columns = ls._report['column_summary']['_columns']
    dropdown1 = widgets.Dropdown(options=numeric_columns,
                                 description='First:')
    dropdown2 = widgets.Dropdown(options=numeric_columns,
                                 description='Second:')
    if len(numeric_columns) > 1:
        dropdown1.value, dropdown2.value = numeric_columns[:2]

    plot_area = widgets.Output()

    for dropdown in [dropdown1, dropdown2]:
        dropdown.observe(lambda x: _update_pairdensity_plot(ls, dropdown1,
                                                            dropdown2,
                                                            plot_area),
                         names='value', type='change')

    _update_pairdensity_plot(ls, dropdown1, dropdown2, plot_area)

    return widgets.VBox([dropdown1, dropdown2, plot_area], padding=PADDING)


def _simple_columnwise_widget(ls, plot_function, columns):
    """Basic column-wise plot widget"""

    dropdown = widgets.Dropdown(options=columns, description='Column:')
    plot_area = widgets.Output()
    update_plot(plot_function, [ls, columns[0]], plot_area, height=PLOT_HEIGHT)

    dropdown.observe(lambda x: update_plot(plot_function, [ls, x['new']],
                                           plot_area, height=PLOT_HEIGHT),
                     names='value', type='change')

    return widgets.VBox([dropdown, plot_area], padding=PADDING)


def create_distribution_plot_widget(ls):
    """Create a distribution plot widget.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.

    Returns
    -------
    :class:`ipywidgets.Widget`
        Jupyter widget to explore distribution plots.
    """
    numeric_columns = ls._report['column_summary']['_columns']
    return _simple_columnwise_widget(ls, plot_distribution, numeric_columns)


def create_cdf_plot_widget(ls):
    """Create a CDF plot widget.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.

    Returns
    -------
    :class:`ipywidgets.Widget`
        Jupyter widget to explore CDF plots.
    """
    numeric_columns = ls._report['column_summary']['_columns']
    return _simple_columnwise_widget(ls, plot_cdf, numeric_columns)


def interactive_explore(ls):
    """Create a widget to visually explore a dataset summary.

    Note that the widget will only work when created within a Jupyter notebook.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.

    Returns
    -------
    :class:`ipywidgets.Widget`
        Jupyter widget with summary plots.
    """
    if not IN_NOTEBOOK:
        message = ('Lens interactive_explore can only be used in a'
                   ' Jupyter notebook')
        logger.error(message)
        raise ValueError(message)
    else:
        # This is a bit of a hack, but it is the only place where the state of
        # plotly initialization is stored. We need to do it because otherwise
        # plotly fails silently if the notebook mode is not initialized.
        if not py.offline.__PLOTLY_OFFLINE_INITIALIZED:
            py.init_notebook_mode()

    tabs = widgets.Tab()
    tabs.children = [create_distribution_plot_widget(ls),
                     create_cdf_plot_widget(ls),
                     create_pairdensity_plot_widget(ls)]
#                     create_correlation_plot_widget(ls)]

    tabs.set_title(0, 'Distribution')
    tabs.set_title(1, 'CDF')
    tabs.set_title(2, 'Pairwise density')
#    tabs.set_title(3, 'Correlation matrix')

    return tabs
