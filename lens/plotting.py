import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
import plotly.figure_factory as pff

DEFAULT_COLORSCALE = "Viridis"


def plot_distribution(ls, column, bins=None):
    """Plot the distribution of numerical columns.

    Create a plotly plot with a histogram of the values in a column. The
    number of bin in the histogram is decided according to the
    Freedman-Diaconis rule unless given by the `bins` parameter.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.
    column : str
        Name of the column.
    bins : int, optional
        Number of bins to use for histogram. If not given, the
        Freedman-Diaconis rule will be used to estimate the best number of
        bins. This argument also accepts the formats taken by the `bins`
        parameter of matplotlib's :function:`~matplotlib.pyplot.hist`.

    Returns
    -------
    :class:`~matplotlib.Axes`
        Matplotlib axes containing the distribution plot.
    """
    column_summary = ls.summary(column)
    if column_summary["notnulls"] <= 2:
        # Plotly refuses to plot histograms if
        # the tdigest has too few values
        raise ValueError(
            "There are fewer than two non-null values in this column"
        )

    if bins is None:
        counts, edges = ls.histogram(column)
    else:
        xs, counts = ls.tdigest_centroids(column)
        counts, edges = np.histogram(xs, weights=counts, bins=bins)

    fig, ax = plt.subplots()

    ax.bar(
        edges[:-1], counts, width=np.diff(edges), label=column, align="edge"
    )

    ax.set_ylim(bottom=0)

    ax.set_xlabel(column)
    ax.set_title('Distribution of column "{}"'.format(column))

    ax.figure.tight_layout()

    return fig


def _set_integer_tick_labels(axis, labels):
    """Use labels dict to set labels on axis"""
    axis.set_major_formatter(FuncFormatter(lambda x, _: labels.get(x, "")))
    axis.set_major_locator(MaxNLocator(integer=True))


def plot_pairdensity_mpl(ls, column1, column2):
    """Plot the pairwise density between two columns.

    This plot is an approximation of a scatterplot through a 2D Kernel
    Density Estimate for two numerical variables. When one of the variables
    is categorical, a 1D KDE for each of the categories is shown,
    normalised to the total number of non-null observations. For two
    categorical variables, the plot produced is a heatmap representation of
    the contingency table.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.
    column1 : str
        First column.
    column2 : str
        Second column.

    Returns
    -------
    :class:`plt.Figure`
        Matplotlib figure containing the pairwise density plot.
    """
    pair_details = ls.pair_details(column1, column2)
    pairdensity = pair_details["pairdensity"]

    x = np.array(pairdensity["x"])
    y = np.array(pairdensity["y"])
    Z = np.array(pairdensity["density"])

    fig, ax = plt.subplots()

    if ls.summary(column1)["desc"] == "categorical":
        idx = np.argsort(x)
        x = x[idx]
        Z = Z[:, idx]
        # Create labels and positions for categorical axis
        x_labels = dict(enumerate(x))
        _set_integer_tick_labels(ax.xaxis, x_labels)
        x = np.arange(-0.5, len(x), 1.0)

    if ls.summary(column2)["desc"] == "categorical":
        idx = np.argsort(y)
        y = y[idx]
        Z = Z[idx]
        y_labels = dict(enumerate(y))
        _set_integer_tick_labels(ax.yaxis, y_labels)
        y = np.arange(-0.5, len(y), 1.0)

    X, Y = np.meshgrid(x, y)

    ax.pcolormesh(X, Y, Z, cmap=DEFAULT_COLORSCALE.lower())

    ax.set_xlabel(column1)
    ax.set_ylabel(column2)

    ax.set_title(r"$\it{{ {} }}$ vs $\it{{ {} }}$".format(column1, column2))

    return fig


def plot_correlation_mpl(ls, include=None, exclude=None):
    """Plot the correlation matrix for numeric columns

    Plot a Spearman rank order correlation coefficient matrix showing the
    correlation between columns. The matrix is reordered to group together
    columns that have a higher correlation coefficient.  The columns to be
    plotted in the correlation plot can be selected through either the
    ``include`` or ``exclude`` keyword arguments. Only one of them can be
    given.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.
    include : list of str
        List of columns to include in the correlation plot.
    exclude : list of str
        List of columns to exclude from the correlation plot.

    Returns
    -------
    :class:`plt.Figure`
        Matplotlib figure containing the pairwise density plot.
    """

    columns, correlation_matrix = ls.correlation_matrix(include, exclude)
    num_cols = len(columns)

    if num_cols > 10:
        annotate = False
    else:
        annotate = True

    fig, ax = plt.subplots()
    sns.heatmap(
        correlation_matrix,
        annot=annotate,
        fmt=".2f",
        ax=ax,
        xticklabels=columns,
        yticklabels=columns,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        square=True,
    )

    ax.xaxis.tick_top()

    # Enforces a width of 2.5 inches per cell in the plot,
    # unless this exceeds 10 inches.
    width_inches = len(columns) * 2.5
    while width_inches > 10:
        width_inches = 10

    fig.set_size_inches(width_inches, width_inches)

    return fig


def plot_cdf(ls, column, N_cdf=100):
    """Plot the empirical cumulative distribution function of a column.

    Creates a plotly plot with the empirical CDF of a column.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.
    column : str
        Name of the column.
    N_cdf : int
        Number of points in the CDF plot.

    Returns
    -------
    :class:`~matplotlib.Axes`
        Matplotlib axes containing the distribution plot.
    """
    tdigest = ls.tdigest(column)

    cdfs = np.linspace(0, 100, N_cdf)
    xs = [tdigest.percentile(p) for p in cdfs]

    fig, ax = plt.subplots()

    ax.set_ylabel("Percentile")
    ax.set_xlabel(column)
    ax.plot(xs, cdfs)

    if ls._report["column_summary"][column]["logtrans"]:
        ax.set_xscale("log")

    ax.set_title("Empirical Cumulative Distribution Function")

    return fig


def plot_pairdensity(ls, column1, column2):
    """Plot the pairwise density between two columns.

    This plot is an approximation of a scatterplot through a 2D Kernel
    Density Estimate for two numerical variables. When one of the variables
    is categorical, a 1D KDE for each of the categories is shown,
    normalised to the total number of non-null observations. For two
    categorical variables, the plot produced is a heatmap representation of
    the contingency table.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.
    column1 : str
        First column.
    column2 : str
        Second column.

    Returns
    -------
    :class:`plotly.Figure`
        Plotly figure containing the pairwise density plot.
    """
    pair_details = ls.pair_details(column1, column2)
    pairdensity = pair_details["pairdensity"]

    x = np.array(pairdensity["x"])
    y = np.array(pairdensity["y"])
    Z = np.array(pairdensity["density"])

    if ls.summary(column1)["desc"] == "categorical":
        idx = np.argsort(x)
        x = x[idx]
        Z = Z[:, idx]

    if ls.summary(column2)["desc"] == "categorical":
        idx = np.argsort(y)
        y = y[idx]
        Z = Z[idx]

    data = [go.Heatmap(z=Z, x=x, y=y, colorscale=DEFAULT_COLORSCALE)]
    layout = go.Layout(title="<i>{}</i> vs <i>{}</i>".format(column1, column2))
    layout["xaxis"] = {
        "type": pairdensity["x_scale"],
        "autorange": True,
        "title": column1,
    }
    layout["yaxis"] = {
        "type": pairdensity["y_scale"],
        "autorange": True,
        "title": column2,
    }
    fig = go.Figure(data=data, layout=layout)
    fig.data[0]["showscale"] = False

    return fig


def plot_correlation(ls, include=None, exclude=None):
    """Plot the correlation matrix for numeric columns

    Plot a Spearman rank order correlation coefficient matrix showing the
    correlation between columns. The matrix is reordered to group together
    columns that have a higher correlation coefficient.  The columns to be
    plotted in the correlation plot can be selected through either the
    ``include`` or ``exclude`` keyword arguments. Only one of them can be
    given.

    Parameters
    ----------
    ls : :class:`~lens.Summary`
        Lens `Summary`.
    include : list of str
        List of columns to include in the correlation plot.
    exclude : list of str
        List of columns to exclude from the correlation plot.

    Returns
    -------
    :class:`plotly.Figure`
        Plotly figure containing the pairwise density plot.
    """

    columns, correlation_matrix = ls.correlation_matrix(include, exclude)
    num_cols = len(columns)

    if num_cols > 10:
        annotate = False
    else:
        annotate = True

    hover_text = []
    for i in range(num_cols):
        hover_text.append(
            [
                "Corr({}, {}) = {:.2g}".format(
                    columns[i], columns[j], correlation_matrix[i, j]
                )
                for j in range(num_cols)
            ]
        )

    if annotate:
        t = np.reshape(
            ["{:.2g}".format(x) for x in correlation_matrix.flatten()],
            correlation_matrix.shape,
        )[::-1].tolist()
    else:
        nrows, ncolumns = correlation_matrix.shape
        t = [["" for i in range(nrows)] for j in range(ncolumns)]

    fig = pff.create_annotated_heatmap(
        z=correlation_matrix.tolist()[::-1],
        colorscale="RdBu",
        x=columns,
        y=columns[::-1],
        zmin=-1.0,
        zmax=1.0,
        annotation_text=t,
        text=hover_text[::-1],
        hoverinfo="text",
    )
    w = len(columns) * 2.5 * 72
    while w > 600:
        w /= np.sqrt(1.4)
    fig.layout["width"] = w
    fig.layout["height"] = w
    fig.data[0]["showscale"] = True

    return fig
