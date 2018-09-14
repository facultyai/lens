"""Summarise a Pandas DataFrame"""

import json
import logging
import os
import time

import numpy as np
import pandas as pd
import scipy

from .dask_graph import create_dask_graph
from .tdigest_utils import tdigest_from_centroids
from .utils import hierarchical_ordering
from .version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class LensSummaryError(Exception):
    pass


class EmptyDataFrameError(Exception):
    pass


def _validate_report(report, schema_version):
    """Validates a dict report"""
    report_schema_version = report.get("_schema_version")
    if (
        report_schema_version is not None
        and report_schema_version != schema_version
    ):
        raise LensSummaryError(
            "The version of the report schema `{}` does "
            "not match the schema version `{}` supported "
            "by this version of lens {}.".format(
                report_schema_version, schema_version, __version__
            )
        )

    columns = report["_columns"]
    column_props = report["column_properties"]
    num_cols = [col for col in columns if (column_props[col]["numeric"])]
    for num_col in num_cols:
        if (
            num_col not in report["column_summary"].keys()
            or num_col not in report["correlation"]["_columns"]
            or num_col not in report["outliers"].keys()
        ):
            raise LensSummaryError(
                "Column `{}` is marked as numeric but "
                "the report lacks its numeric summary"
                " and correlation".format(num_col)
            )

    cat_cols = [col for col in columns if column_props[col]["is_categorical"]]
    for cat_col in cat_cols:
        if cat_col not in report["frequencies"].keys():
            raise LensSummaryError(
                "Column `{}` is marked as categorical but "
                "the report lacks its frequency analysis".format(cat_col)
            )


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class Summary(object):
    """A summary of a pandas DataFrame.

    Create a summary instance by calling :func:`lens.summarise.summarise` on a
    DataFrame.  This calculates several quantities of interest to data
    scientists.

    The Summary object is designed for programmatic use. For more direct
    visual inspection, use the :class:`lens.explorer.Explorer` class
    in a Jupyter notebook.

    """

    schema_version = 1

    def __init__(self, report):
        if not isinstance(report, dict):
            raise TypeError("report argument must be a dict")

        if "_schema_version" not in report.keys():
            report["_schema_version"] = self.schema_version

        _validate_report(report, schema_version=self.schema_version)
        self._report = report

    @staticmethod
    def from_json(file):
        """Create a Summary from a report saved in JSON format.

        Parameters
        ----------
        file : str or buffer
            Path to file containing the JSON report or buffer from which the
            report can be read.

        Returns
        -------
        :class:`~lens.summarise.Summary`
            ``Summary`` object containing the summary in the JSON file.
        """
        if hasattr(file, "read"):
            report = json.load(file)
        else:
            with open(file, "r") as f:
                report = json.load(f)

        return Summary(report)

    def to_json(self, file=None):
        """Produce a JSON serialization of the report.

        Parameters
        ----------
        file : str or buffer, optional
            File name or writeable buffer to save the JSON report. If omitted,
            a string containing the report will be returned.

        Returns
        -------
        str
           JSON serialization of the summary report
        """
        if file is None:
            return json.dumps(
                self._report, separators=(",", ":"), cls=NumpyEncoder
            )
        else:
            if hasattr(file, "write"):
                json.dump(
                    self._report, file, separators=(",", ":"), cls=NumpyEncoder
                )
            else:
                with open(file, "w") as f:
                    json.dump(
                        self._report,
                        f,
                        separators=(",", ":"),
                        cls=NumpyEncoder,
                    )

    @property
    def columns(self):
        """Get a list of column names of the dataset.

        Returns
        -------
        list
            Column names

        Examples
        --------

        >>> summary.columns
        ['fixed acidity',
         'volatile acidity',
         'citric acid',
         'residual sugar',
         'chlorides',
         'free sulfur dioxide',
         'total sulfur dioxide',
         'density',
         'pH',
         'sulphates',
         'alcohol',
         'quality']
        """
        return self._report["_columns"]

    @property
    def rows(self):
        """Get the number of rows in the dataset.

        Returns
        -------
        int
            Number of rows

        Examples
        --------

        >>> summary.rows
        4898
        """
        return self._report["row_count"]["total"]

    @property
    def rows_unique(self):
        """Get the number of unique rows in the dataset.

        Returns
        -------
        int
            Number of unique rows.
        """
        return self._report["row_count"]["unique"]

    def _desc(self, column):
        """Return the inferred description of a column.

        Parameters
        ----------
        column : str
            Column name.

        Returns
        -------
        str
            Description of the column.
        """

        column_props = self._report["column_properties"][column]

        if column_props["is_categorical"]:
            return "categorical"
        elif column_props["numeric"]:
            return "numeric"
        elif column_props["is_ID"]:
            return "ID_like"
        else:
            return None

    def summary(self, column):
        """Basic information about the column

        This returns information about the number of nulls and unique
        values in ``column`` as well as which type this column is.
        This is guaranteed to return a dictionary with the same keys
        for every column.

        The dictionary contains the following keys:

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


        Examples
        --------

        >>> summary.summary('quality')
        {'desc': 'categorical',
         'dtype': 'int64',
         'name': 'quality',
         'notnulls': 4898,
         'nulls': 0,
         'unique': 7}

        >>> summary.summary('chlorides')
        {'desc': 'numeric',
         'dtype': 'float64',
         'name': 'chlorides',
         'notnulls': 4898,
         'nulls': 0,
         'unique': 160}

        Parameters
        ----------
        column : str
            Column name

        Returns
        -------
        dict
            Dictionary of summary information.
        """
        if column not in self._report["_columns"]:
            raise LensSummaryError(
                "The data summary does not contain"
                " information about column `{}`.".format(column)
            )

        column_props = self._report["column_properties"][column]

        summary = {"name": column, "desc": self._desc(column)}

        for key in ["nulls", "notnulls", "unique", "dtype"]:
            summary[key] = column_props[key]

        return summary

    def details(self, column):
        """Type-specific information for a column

        The `details` method returns additional information on ``column``,
        beyond that provided by the ``summary`` method. If ``column`` is
        numeric, this returns summary statistics. If it is categorical,
        it returns a dictionary of how often each category occurs.

        Examples
        --------

        >>> summary.details('alcohol')
        {'desc': 'numeric',
         'iqr': 1.9000000000000004,
         'max': 14.199999999999999,
         'mean': 10.514267047774602,
         'median': 10.4,
         'min': 8.0,
         'name': 'alcohol',
         'std': 1.2306205677573181,
         'sum': 51498.880000000005}

        >>> summary.details('quality')
        {'desc': 'categorical',
         'frequencies':
              {3: 20, 4: 163, 5: 1457, 6: 2198, 7: 880, 8: 175, 9: 5},
         'iqr': 1.0,
         'max': 9,
         'mean': 5.8779093507554103,
         'median': 6.0,
         'min': 3,
         'name': 'quality',
         'std': 0.88563857496783116,
         'sum': 28790}

        Parameters
        ----------
        column : str
            Column name

        Returns
        -------
        dict
            Dictionary of detailed information.
        """
        if column not in self._report["_columns"]:
            raise LensSummaryError(
                "The data summary does not contain"
                " information about column `{}`.".format(column)
            )

        column_props = self._report["column_properties"][column]

        details = {"name": column, "desc": self._desc(column)}

        if column_props["is_categorical"]:
            details["frequencies"] = self._report["frequencies"][column]

        if column_props["numeric"]:
            column_summ = self._report["column_summary"][column]
            for k in ["min", "max", "mean", "median", "std", "sum", "iqr"]:
                details[k] = column_summ[k]
        return details

    def pair_details(self, first, second):
        """Get pairwise information for a column pair.

        The information returned depends on the types of the two columns.
        It may contain the following keys.

        correlation
            dictionary with the Spearman rank correlation
            coefficient and Pearson product-moment correlation coefficient
            between the columns. This is returned when both columns are
            numeric.

        pairdensity
            dictionary with an estimate of the pairwise
            density between the columns. The density is either
            a 2D KDE estimate if both columns are numerical, or
            several 1D KDE estimates if one of the columns is categorical
            and the other numerical (grouped by the categorical column)
            or a cross-tabuluation.

        Examples
        --------

        >>> summary.pair_details('chlorides', 'quality')
        {'correlation': {
            'pearson': -0.20993441094675602,
            'spearman': -0.31448847828244203},
        {'pairdensity': {
            'density': <2d numpy array>
            'x': <1d numpy array of x-values>
            'y': <1d numpy array of y-values>
            'x_scale': 'linear',
            'y_scale': 'cat'}
        }

        >>> summary.pair_details('alcohol', 'chlorides')
        {'correlation': {
            'pearson': -0.36018871210816106,
            'spearman': -0.5708064071153713},
        {'pairdensity': {
            'density': <2d numpy array>
            'x': <1d numpy array of x-values>
            'y': <1d numpy array of y-values>
            'x_scale': 'linear',
            'y_scale': 'linear'}
        }

        Parameters
        ----------
        first : str
            Name of the first column.
        second : str
            Name of the second column.

        Returns
        -------
        dict
            Dictionary of pairwise information.
        """
        if first == second:
            raise ValueError(
                "Can only return the pair details of two different columns: "
                "received {} twice.".format(first)
            )

        pair_details = {}

        # Correlation

        corr_report = self._report["correlation"]
        try:
            idx = [
                corr_report["_columns"].index(col) for col in [first, second]
            ]
        except ValueError as e:
            logger.info(
                "No correlation information for column `{}`".format(
                    e.args[0].split()[0]
                )
            )
        else:
            correlation = {
                k: corr_report[k][idx[0]][idx[1]]
                for k in ["spearman", "pearson"]
            }
            pair_details["correlation"] = correlation

        # Pair density / Crosstab

        pairdensity_report = self._report["pairdensity"]

        # We store pairdensity information for both first/second and
        # second/first in a single key in the report, so we check for both
        # report[first][second] and report[second][first] to find it and
        # transpose if necessary.
        try:
            pairdensity = pairdensity_report[first][second]
            scales = pairdensity["scales"]
            density = np.array(pairdensity["density"])
        except KeyError:
            try:
                pairdensity = pairdensity_report[second][first]
                # Invert scale information and transpose matrix
                scales = pairdensity["scales"][::-1]
                density = np.array(pairdensity["density"]).T
            except KeyError:
                logger.info(
                    "No pairdensity information for columns `{}`"
                    " and `{}`".format(first, second)
                )
                pairdensity = None

        if pairdensity is not None:
            pairdensity = {
                "density": density,
                "x": pairdensity["axes"][first],
                "y": pairdensity["axes"][second],
                "x_scale": scales[0],
                "y_scale": scales[1],
            }

            pair_details["pairdensity"] = pairdensity

        return pair_details

    def histogram(self, column):
        """
        Return the histogram for `column`.

        This function returns a histogram for the column. The number of bins is
        estimated through the Freedman-Diaconis rule.

        Parameters
        ----------

        column: str
            Name of the column

        Returns
        -------

        counts: array
            Counts for each of the bins of the histogram.
        bin_edges : array
            Edges of the bins in the histogram. Length is ``length(counts)+1``.
        """
        self._check_column_name(column)
        try:
            histogram = self._report["column_summary"][column]["histogram"]
        except KeyError:
            raise ValueError("{} is not a numeric column".format(column))

        return [np.array(histogram[key]) for key in ["counts", "bin_edges"]]

    def kde(self, column):
        """
        Return a Kernel Density Estimate for `column`.

        This function returns a KDE for the column. It is computed between the
        minimum and maximum values of the column and uses Scott's rule to
        compute the bandwith.

        Parameters
        ----------

        column: str
            Name of the column

        Returns
        -------

        x: array
            Values at which the KDE has been evaluated.
        y : array
            Values of the KDE.
        """
        self._check_column_name(column)
        try:
            kde = self._report["column_summary"][column]["kde"]
        except KeyError:
            raise ValueError("{} is not a numeric column".format(column))

        return [np.array(kde[key]) for key in ["x", "y"]]

    def _tdigest_report(self, column):
        """ Return the list of tdigest centroids and means from report
        """
        self._check_column_name(column)
        try:
            tdigest_list = self._report["column_summary"][column]["tdigest"]
        except KeyError:
            raise ValueError("{} is not a numeric column".format(column))
        return tdigest_list

    def tdigest_centroids(self, column):
        """Get TDigest centroids and counts for column.

        Parameters
        ----------
        column : str
            Name of the column.

        Returns
        -------
        :class:`numpy.array`
            Means of the TDigest centroids.
        :class:`numpy.array`
            Counts for each of the TDigest centroids.
        """

        tdigest_list = self._tdigest_report(column)
        xs, counts = zip(*tdigest_list)
        return np.array(xs), np.array(counts)

    def pdf(self, column):
        """ Approximate pdf for `column`

        This returns a function representing the pdf of a numeric column.

        Examples
        --------

        >>> pdf = summary.pdf('chlorides')
        >>> min_value = summary.details('chlorides')['min']
        >>> max_value = summary.details('chlorides')['max']
        >>> xs = np.linspace(min_value, max_value, 200)
        >>> plt.plot(xs, pdf(xs))

        Parameters
        ----------

        column : str
            Name of the column.

        Returns
        -------
        pdf: function
            Function representing the pdf.
        """
        xs, counts = self.tdigest_centroids(column)
        return scipy.interpolate.interp1d(xs, counts)

    def tdigest(self, column):
        """Return a TDigest object approximating the distribution of a column

        Documentation for the TDigest class can be found at
        https://github.com/CamDavidsonPilon/tdigest.

        Parameters
        ----------
        column : str
            Name of the column.

        Returns
        -------
        :class:`tdigest.TDigest`
            TDigest instance computed from the values of the column.
        """
        return tdigest_from_centroids(self._tdigest_report(column))

    def cdf(self, column):
        """ Approximate cdf for `column`

        This returns a function representing the cdf of a numeric column.

        Examples
        --------

        >>> cdf = summary.cdf('chlorides')
        >>> min_value = summary.details('chlorides')['min']
        >>> max_value = summary.details('chlorides')['max']
        >>> xs = np.linspace(min_value, max_value, 200)
        >>> plt.plot(xs, cdf(xs))

        Parameters
        ----------

        column : str
            Name of the column.

        Returns
        -------
        cdf: function
            Function representing the cdf.
        """
        tdigest = self.tdigest(column)
        return tdigest.cdf

    def correlation_matrix(self, include=None, exclude=None):
        """ Correlation matrix for numeric columns

        Parameters
        ----------

        include: list of strings, optional
            List of numeric columns to include. Includes all columns
            by default.

        exclude: list of strings, optional
            List of numeric columns to exclude. Includes all columns
            by default.

        Returns
        -------

        columns: list of strings
            List of column names

        correlation_matrix: 2D array of floats
            The correlation matrix, ordered such that
            ``correlation_matrix[i, j]`` is the correlation between
            ``columns[i]`` and ``columns[j]``

        Notes
        -----

        The columns are ordered through hierarchical clustering. Thus,
        neighbouring columns in the output will be more correlated.
        """
        if include is not None and exclude is not None:
            raise ValueError(
                "Either 'include' or 'exclude' should be defined, "
                "but not both"
            )

        available_columns = self._report["correlation"]["_columns"]
        if include is not None:
            non_numeric_includes = set(include) - set(available_columns)
            if non_numeric_includes:
                raise ValueError(
                    "Only numeric columns can be included in the "
                    "correlation plot. Columns {} are not "
                    "numeric".format(non_numeric_includes)
                )
            columns = include
        elif exclude is not None:
            columns = set(available_columns) - set(exclude)
        else:
            columns = available_columns
        columns = list(columns)

        # Filter the correlation matrix to select only the above columns
        correlation_report = self._report["correlation"]
        idx = [correlation_report["_columns"].index(col) for col in columns]
        correlation_matrix = np.array(correlation_report["spearman"])[idx][
            :, idx
        ]

        return hierarchical_ordering(columns, correlation_matrix)

    def _check_column_name(self, column):
        if column not in self.columns:
            raise KeyError(column)


def summarise(
    df,
    scheduler="multiprocessing",
    num_workers=None,
    size=None,
    pairdensities=True,
):
    """Create a Lens Summary for a Pandas DataFrame.

    This creates a :class:`~lens.Summary` instance containing
    many quantities of interest to a data scientist.

    Examples
    --------

    Let's explore the wine quality dataset.

    >>> import pandas as pd
    >>> import lens
    >>> url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"  # noqa
    >>> wines_df = pd.read_csv(url, sep=';')
    >>> summary = lens.summarise(wines_df)

    Now that we have a :class:`~lens.Summary` instance we can inspect
    the shape of the dataset

    >>> summary.columns
    ['fixed acidity',
     'volatile acidity',
     'citric acid',
     'residual sugar',
     'chlorides',
     'free sulfur dioxide',
     'total sulfur dioxide',
     'density',
     'pH',
     'sulphates',
     'alcohol',
     'quality']
    >>> summary.rows
    4898

    So far, nothing groundbreaking. Let's look at the ``quality`` column:

    >>> summary.summary('quality')
    {'desc': 'categorical',
     'dtype': 'int64',
     'name': 'quality',
     'notnulls': 4898,
     'nulls': 0,
     'unique': 7}

    This tells us that there are seven unique values in the quality columns,
    and zero null values. It also tells us that lens will treat this
    column as categorical. Let's look at this in more details:

    >>> summary.details('quality')
    {'desc': 'categorical',
     'frequencies': {3: 20, 4: 163, 5: 1457, 6: 2198, 7: 880, 8: 175, 9: 5},
     'iqr': 1.0,
     'max': 9,
     'mean': 5.8779093507554103,
     'median': 6.0,
     'min': 3,
     'name': 'quality',
     'std': 0.88563857496783116,
     'sum': 28790}

    This tells us that the median wine quality is 6 and the standard deviation
    is less than one. Let's now get the correlation between the ``quality``
    column and the ``alcohol`` column:

    >>> summary.pair_detail('quality', 'alcohol')['correlation']
    {'pearson': 0.4355747154613688, 'spearman': 0.4403691816246831}

    Thus, the Spearman Rank Correlation coefficient between these two columns
    is 0.44.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be analysed.
    scheduler : str, optional
        Dask scheduler to use. Must be one of [distributed, multiprocessing,
        processes, single-threaded, sync, synchronous, threading, threads].
    num_workers : int or None, optional
        Number of workers in the pool. If the environment variable `NUM_CPUS`
        is set that number will be used, otherwise it will use as many workers
        as CPUs available in the machine.
    size : int, optional
        DataFrame size on disk, which will be added to the report.
    pairdensities : bool, optional
        Whether to compute the pairdensity estimation between all pairs of
        numerical columns. For most datasets, this is the most expensive
        computation. Default is True.

    Returns
    -------
    summary : :class:`~lens.Summary`
        The computed data summary.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Can only summarise a Pandas DataFrame")

    if len(df.columns) == 0:
        raise EmptyDataFrameError("The DataFrame has no columns")

    if num_workers is None:
        try:
            num_workers = int(os.environ["NUM_CPUS"])
            logger.debug(
                "Number of workers read from environment: {}".format(
                    num_workers
                )
            )
        except ValueError:
            # Set to None if NUM_CPUS cannot be cast to an integer
            logger.warning(
                "Environment variable NUM_CPUS={} cannot be"
                " interpreted as an integer, defaulting to"
                " number of cores in system".format(os.environ.get("NUM_CPUS"))
            )
            num_workers = None
        except KeyError:
            # NUM_CPUS not in environment
            num_workers = None

    kwargs = {"scheduler": scheduler}
    if num_workers is not None:
        kwargs["num_workers"] = num_workers

    tstart = time.time()
    report = create_dask_graph(df, pairdensities=pairdensities).compute(
        **kwargs
    )
    report["_run_time"] = time.time() - tstart

    report["_lens_version"] = __version__

    if size is not None:
        report["size"] = size

    return Summary(report)
