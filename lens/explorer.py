"""Explore a Summary"""

import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

import plotly.tools
import plotly.offline as py

from lens.summarise import Summary
from lens.formatting import JupyterTable
from lens.plotting import (
    plot_distribution,
    plot_pairdensity,
    plot_correlation,
    plot_cdf,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Check whether we are in a notebook environment
# this is a false positive if we are in the Jupyter console
IN_NOTEBOOK = "ipykernel" in sys.modules

PLOTLY_TO_MPL_KWS = {"strip_style": True, "resize": True}

PLOTLY_KWS = {"show_link": False}


def _render(fig, showlegend=None):
    """Plot a matploltib or plotly figure"""
    if isinstance(fig, plt.Figure):
        fig = plotly.tools.mpl_to_plotly(fig, **PLOTLY_TO_MPL_KWS)

    if showlegend is not None:
        fig.layout["showlegend"] = showlegend

    if not IN_NOTEBOOK:
        message = "Lens explorer can only plot in a Jupyter notebook"
        logger.error(message)
        raise ValueError(message)
    else:
        if not py.offline.__PLOTLY_OFFLINE_INITIALIZED:
            py.init_notebook_mode()
        return py.iplot(fig, **PLOTLY_KWS)


class Explorer(object):
    """An explorer to visualise a Lens Summary

    Once a Lens ``Summary`` has been generated with
    :func:`lens.summarise.summarise`, this class provides the methods necessary
    to explore the summary though tables and plots. It is best used from within
    a Jupyter notebook.
    """

    # Number of points to show in the CDF plot
    _N_cdf = 1000

    def __init__(self, summary, plot_renderer=_render):
        if not isinstance(summary, Summary):
            raise TypeError("Can only explore a lens Summary")
        self.summary = summary
        self.plot_renderer = plot_renderer

    def describe(self):
        """General description of the dataset.

        Produces a table including the following information about each column:

        ``desc``
            the type of data: currently ``categorical`` or ``numeric``.
            Lens will calculate different quantities for this column
            depending on the value of ``desc``.

        ``dtype``
            the type of data in Pandas.

        ``name``
            column name

        ``notnulls``
            number of non-null values in the column

        ``nulls``
            number of null-values in the column

        ``unique``
            number of unique values in the column
        """
        summary = self.summary
        columns = summary.columns

        header = [""]
        header.extend(columns)

        desc = ["desc"]
        desc.extend([summary._desc(column) for column in columns])

        dtype = ["dtype"]
        dtype.extend([summary.summary(column)["dtype"] for column in columns])

        notnulls = ["notnulls"]
        notnulls.extend(
            [summary.summary(column)["notnulls"] for column in columns]
        )

        nulls = ["nulls"]
        nulls.extend([summary.summary(column)["nulls"] for column in columns])

        unique = ["unique"]
        unique.extend(
            [summary.summary(column)["unique"] for column in columns]
        )

        return JupyterTable([header, desc, dtype, notnulls, nulls, unique])

    def column_details(self, column, sort=False):
        """Show type-specific column details.

        For numeric columns, this method produces a table with summary
        statistics, including minimum, maximum, mean, and median. For
        categorical columns, it produces a frequency table for each category
        sorted in descending order of frequency.

        Parameters
        ----------
        column : str
            Name of the column.
        sort : boolean, optional
            Sort frequency tables in categorical variables by
            category name.
        """
        details = self.summary.details(column)
        desc = details["desc"]

        if desc == "numeric":
            caption = ""
            data = [
                ["", details["name"]],
                ["desc", details["desc"]],
                ["dtype", self.summary.summary(column)["dtype"]],
                ["min", details["min"]],
                ["max", details["max"]],
                ["mean", details["mean"]],
                ["median", details["median"]],
                ["std", details["std"]],
                ["sum", details["sum"]],
                ["IQR", details["iqr"]],
            ]
            return JupyterTable(data)
        elif desc == "categorical":
            caption = "<p>desc: {}, dtype: {}</p>".format(
                details["desc"], self.summary.summary(column)["dtype"]
            )
            data = [["item", "frequency"]]
            frequencies = []
            for item, frequency in details["frequencies"].items():
                frequencies.append([item, frequency])
            if sort:
                data.extend(sorted(frequencies, key=lambda x: x[0]))
            else:
                data.extend(sorted(frequencies, key=lambda x: -x[1]))
        else:
            caption = ""
            data = [
                ["", details["name"]],
                ["desc", details["desc"]],
                ["dtype", self.summary.summary(column)["dtype"]],
            ]

        return JupyterTable(data, caption=caption)

    def distribution(self, column):
        """Show properties of the distribution of values in the column.

        Parameters
        ----------
        column : str
            Name of the column.
        """
        raise NotImplementedError

    def distribution_plot(self, column, bins=None):
        """Plot the distribution of a numeric column.

        Create a plotly plot with a histogram of the values in a column. The
        number of bin in the histogram is decided according to the
        Freedman-Diaconis rule unless given by the `bins` parameter.

        Parameters
        ----------
        column : str
            Name of the column.
        bins : int, optional
            Number of bins to use for histogram. If not given, the
            Freedman-Diaconis rule will be used to estimate the best number of
            bins. This argument also accepts the formats taken by the `bins`
            parameter of matplotlib's :function:`~matplotlib.pyplot.hist`.
        """
        ax = plot_distribution(self.summary, column, bins)
        self.plot_renderer(ax)

    def cdf_plot(self, column):
        """Plot the empirical cumulative distribution function of a column.

        Creates a plotly plot with the empirical CDF of a column.

        Parameters
        ----------
        column : str
            Name of the column.
        """
        ax = plot_cdf(self.summary, column, self._N_cdf)
        self.plot_renderer(ax)

    def crosstab(self, column1, column2):
        """Show a contingency table of two categorical columns.

        Print a contingency table for two categorical variables showing the
        multivariate frequancy distribution of the columns.

        Parameters
        ----------
        column1 : str
            First column.
        column2 : str
            Second column.
        """
        pair_details = self.summary.pair_details(column1, column2)

        for column in [column1, column2]:
            column_details = self.summary.details(column)
            if column_details["desc"] != "categorical":
                raise ValueError(
                    "Column `{}` is not categorical".format(column)
                )

        pair_details = self.summary.pair_details(column1, column2)
        pairdensity = pair_details["pairdensity"]

        # Convert to numpy arrays for ease of reindexing
        x = np.array(pairdensity["x"])
        y = np.array(pairdensity["y"])
        crosstab = np.array(pairdensity["density"])

        # Sort by first column category names
        idx = np.argsort(x)
        x = x[idx]
        crosstab = crosstab[:, idx]

        # Sort by second column category names
        idx = np.argsort(y)
        y = y[idx]
        crosstab = crosstab[idx]

        table = [[""] + x.tolist()]
        for y_category, crosstab_row in zip(y, crosstab):
            table.append([y_category] + crosstab_row.tolist())

        return JupyterTable(table)

    def pairwise_density_plot(self, column1, column2):
        """Plot the pairwise density between two columns.

        This plot is an approximation of a scatterplot through a 2D Kernel
        Density Estimate for two numerical variables. When one of the variables
        is categorical, a 1D KDE for each of the categories is shown,
        normalised to the total number of non-null observations. For two
        categorical variables, the plot produced is a heatmap representation of
        the contingency table.

        Parameters
        ----------
        column1 : str
            First column.
        column2 : str
            Second column.
        """
        allowed_descriptions = ["numeric", "categorical"]
        for column in [column1, column2]:
            column_description = self.summary.summary(column)["desc"]
            if column_description not in allowed_descriptions:
                raise ValueError(
                    "Column {} is not numeric or categorical".format(column)
                )

        fig = plot_pairdensity(self.summary, column1, column2)
        self.plot_renderer(fig)

    def correlation_plot(self, include=None, exclude=None):
        """Plot the correlation matrix for numeric columns

        Plot a Spearman rank order correlation coefficient matrix showing the
        correlation between columns. The matrix is reordered to group together
        columns that have a higher correlation coefficient.  The columns to be
        plotted in the correlation plot can be selected through either the
        ``include`` or ``exclude`` keyword arguments. Only one of them can be
        given.

        Parameters
        ----------

        include : list of str
            List of columns to include in the correlation plot.
        exclude : list of str
            List of columns to exclude from the correlation plot.
        """
        fig = plot_correlation(self.summary, include, exclude)
        self.plot_renderer(fig)

    def correlation(self, include=None, exclude=None):
        """Show the correlation matrix for numeric columns.

        Print a Spearman rank order correlation coefficient matrix in tabular
        form, showing the correlation between columns. The matrix is reordered
        to group together columns that have a higher correlation coefficient.
        The columns to be shown in the table can be selected
        through either the ``include`` or ``exclude`` keyword arguments. Only
        one of them can be given.

        Parameters
        ----------

        include : list of str
            List of columns to include in the correlation plot.
        exclude : list of str
            List of columns to exclude from the correlation plot.
        """
        columns, correlation_matrix = self.summary.correlation_matrix(
            include, exclude
        )
        headers = [""] + columns
        rows = []
        for column, correlation_row in zip(columns, correlation_matrix):
            rows.append([column] + correlation_row.tolist())
        return JupyterTable([headers] + rows)


def explore(summary):
    """Create an Explorer instance from a Lens Summary"""
    return Explorer(summary)
