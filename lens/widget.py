from __future__ import division
import sys
import logging
from ipywidgets import widgets
from IPython.display import display
from lens.plotting import (
    plot_distribution,
    plot_cdf,
    plot_pairdensity_mpl,
    plot_correlation_mpl,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Check whether we are in a notebook environment
# this is a false positive if we are in the Jupyter console
IN_NOTEBOOK = "ipykernel" in sys.modules

PADDING = "10px"
PLOT_HEIGHT = 400
PLOT_WIDTH = 600
DPI = 72


def update_plot(f, args, plot_area, **kwargs):
    """Updates the content of an output widget with rendered function"""

    fig = f(*args)
    plot_area.clear_output()

    height = kwargs.get("height", PLOT_HEIGHT)
    width = kwargs.get("width", PLOT_WIDTH)
    dpi = kwargs.get("dpi", DPI)

    fig.set_size_inches(width / dpi, height / dpi)

    plot_area.layout.height = "{:.0f}px".format(height)
    plot_area.layout.width = "{:.0f}px".format(width)

    with plot_area:
        display(fig)


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

    plot_area = widgets.Output()

    update_plot(
        plot_correlation_mpl,
        [ls],
        plot_area,
        height=PLOT_WIDTH,
        width=PLOT_WIDTH * 1.3,
    )

    return plot_area


def _update_pairdensity_plot(ls, dd1, dd2, plot_area):
    if dd1.value != dd2.value:
        update_plot(
            plot_pairdensity_mpl,
            [ls, dd1.value, dd2.value],
            plot_area,
            height=600,
            width=600,
        )


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
    numeric_columns = ls._report["column_summary"]["_columns"]
    dropdown1 = widgets.Dropdown(options=numeric_columns, description="First:")
    dropdown2 = widgets.Dropdown(
        options=numeric_columns, description="Second:"
    )
    if len(numeric_columns) > 1:
        dropdown1.value, dropdown2.value = numeric_columns[:2]

    plot_area = widgets.Output()

    for dropdown in [dropdown1, dropdown2]:
        dropdown.observe(
            lambda x: _update_pairdensity_plot(
                ls, dropdown1, dropdown2, plot_area
            ),
            names="value",
            type="change",
        )

    _update_pairdensity_plot(ls, dropdown1, dropdown2, plot_area)

    return widgets.VBox([dropdown1, dropdown2, plot_area], padding=PADDING)


def _simple_columnwise_widget(ls, plot_function, columns):
    """Basic column-wise plot widget"""

    dropdown = widgets.Dropdown(options=columns, description="Column:")
    plot_area = widgets.Output()
    update_plot(plot_function, [ls, columns[0]], plot_area, height=PLOT_HEIGHT)

    dropdown.observe(
        lambda x: update_plot(
            plot_function, [ls, x["new"]], plot_area, height=PLOT_HEIGHT
        ),
        names="value",
        type="change",
    )

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
    numeric_columns = ls._report["column_summary"]["_columns"]
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
    numeric_columns = ls._report["column_summary"]["_columns"]
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
        message = (
            "Lens interactive_explore can only be used in a"
            " Jupyter notebook"
        )
        logger.error(message)
        raise ValueError(message)

    tabs = widgets.Tab()
    tabs.children = [
        create_distribution_plot_widget(ls),
        create_cdf_plot_widget(ls),
        create_pairdensity_plot_widget(ls),
        create_correlation_plot_widget(ls),
    ]

    tabs.set_title(0, "Distribution")
    tabs.set_title(1, "CDF")
    tabs.set_title(2, "Pairwise density")
    tabs.set_title(3, "Correlation matrix")

    return tabs
