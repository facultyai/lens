"""Metrics for the computation of a Lens summary"""

from __future__ import division

import logging
import time
from functools import wraps

from tdigest import TDigest
import numpy as np
from scipy import stats
from scipy import signal
import pandas as pd

from .utils import hierarchical_ordering_indices

DENSITY_N = 100
LOGNORMALITY_P_THRESH = 0.05
CAT_FRAC_THRESHOLD = 0.5

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def timeit(func):
    """Decorator to time callable execution and add it to the report.

    Parameters
    ----------
    func : callable
        The callable to execute.

    Returns
    -------
    callable
        Decorated function.
    """

    @wraps(func)
    def decorator(*args, **kwargs):
        tstart = time.time()
        report = func(*args, **kwargs)
        if report is not None:
            report["_run_time"] = time.time() - tstart
        return report

    return decorator


@timeit
def row_count(df):
    """Count number of total and unique rows.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame.

    Returns
    -------
    dict
        Dictionary with `total` and `unique` keys.
    """
    report = {}
    report["total"] = len(df.index)
    report["unique"] = len(df.drop_duplicates().index)
    return report


@timeit
def column_properties(series):
    """Infer properties of a Pandas Series.

    Parameters
    ----------
    series : pd.Series
        Series to infer properties of.

    Returns
    -------
    dict
        Dictionary of inferred properties.
    """
    cat_N_threshold = {"object": 1000, "int64": 10, "float64": 10}

    name = series.name
    colresult = {}
    colresult["dtype"] = str(series.dtype)
    nulls = series.isnull().sum()
    colresult["nulls"] = int(nulls) if not np.isnan(nulls) else 0
    notnulls = series.dropna()

    colresult["notnulls"] = len(notnulls.index)
    colresult["numeric"] = (
        series.dtype in [np.float64, np.int64] and colresult["notnulls"] > 0
    )
    unique = notnulls.unique().size
    colresult["unique"] = unique
    colresult["is_categorical"] = False
    if (
        colresult["dtype"] in {"object", "int64", "float64"}
        and colresult["notnulls"] > 0
    ):
        # In Pandas integers with nulls are cast as floats, so we have
        # to include floats as possible categoricals to detect
        # categorical integers.
        colresult["is_categorical"] = (
            unique / colresult["notnulls"] <= CAT_FRAC_THRESHOLD
        ) and (unique <= cat_N_threshold[colresult["dtype"]])
        logger.debug(
            "Column {:15}: {:6} unique, {:6} notnulls, {:6} total"
            " --> {}categorical".format(
                name,
                unique,
                colresult["notnulls"],
                colresult["notnulls"] + colresult["nulls"],
                "NOT " * (not colresult["is_categorical"]),
            )
        )

    # Don't use the is_ID field for now:
    # it's too prone to false positives.
    # If a columns is wrongly identified as ID-like,
    # it doesn't get analyzed
    colresult["is_ID"] = False

    return {name: colresult, "_columns": [name]}


def _tdigest_mean(digest):
    """TODO

    Parameters
    ----------
    digest : tdigest.TDigest
        t-digest data structure.

    Returns
    -------
    TODO
    """
    means = [c.mean for c in digest.C.values()]
    counts = [c.count for c in digest.C.values()]
    return np.average(means, weights=counts)


def _tdigest_std(digest):
    """TODO

    Parameters
    ----------
    digest : tdigest.TDigest
        t-digest data structure.

    Returns
    -------
    TODO
    """
    mean = _tdigest_mean(digest)
    sums = [(x.mean - mean) ** 2 * x.count for x in digest.C.values()]
    return np.sqrt(np.sum(sums) / digest.n)


def _tdigest_normalise(digest):
    """TODO

    Parameters
    ----------
    digest : tdigest.TDigest
        t-digest data structure.

    Returns
    -------
    TODO
    """
    m = _tdigest_mean(digest)
    s = _tdigest_std(digest)
    ndigest = TDigest()
    for x in digest.C.values():
        ndigest.update((x.mean - m) / s, x.count)
    return ndigest


def _tdigest_norm_kstest(digest):
    """TODO

    Parameters
    ----------
    digest : tdigest.TDigest
        t-digest data structure.

    Returns
    -------
    TODO
    """
    normdigest = _tdigest_normalise(digest)

    x = np.linspace(-3, 3, 500)
    dig_q = np.array([normdigest.cdf(xx) for xx in x])
    norm_q = stats.norm.cdf(x)

    D = np.max(np.abs(dig_q - norm_q))

    if digest.n > 3000:
        return D, stats.distributions.kstwobign.sf(D * np.sqrt(digest.n))
    else:
        return D, 2 * stats.distributions.ksone.sf(D, digest.n)


def _test_logtrans(digest):
    """
    Test if t-digest distribution is more normal when log-transformed.

    Test whether a log-transform improves normality of data with a
    simplified Kolmogorov-Smirnov two-sided test (the location and scale
    of the normal distribution are estimated from the median and
    standard deviation of the data).

    Parameters
    ----------
    digest : tdigest.TDigest
        t-digest data structure.

    Returns
    -------
    TODO
    """
    if digest.percentile(0) <= 0:
        return False

    logdigest = TDigest()
    for c in digest.C.values():
        logdigest.update(np.log(c.mean), c.count)

    lKS, lp = _tdigest_norm_kstest(logdigest)
    KS, p = _tdigest_norm_kstest(digest)
    logger.debug(
        "KSnorm: log: {:.2g}, {:.2g}; linear: {:.2g}, {:.2g}".format(
            lKS, lp, KS, p
        )
    )

    return (
        (lKS < KS)
        and (lp > p)
        and (lp > LOGNORMALITY_P_THRESH)
        and (p < LOGNORMALITY_P_THRESH)
    )


@timeit
def column_summary(series, column_props, delta=0.01):
    """Summarise a numeric column.

    Parameters
    ----------
    series : pd.Series
        Numeric column.
    column_props : TODO
        TODO
    delta : float
        TODO

    Returns
    -------
    TODO
    """
    col = series.name
    if not column_props[col]["numeric"] or column_props[col]["notnulls"] == 0:
        # Series is not numeric or is all NaNs.
        return None

    logger.debug("column_summary - " + col)

    # select non-nulls from column
    data = series.dropna()

    colresult = {}
    for m in ["mean", "min", "max", "std", "sum"]:
        val = getattr(data, m)()
        if type(val) is np.int64:
            colresult[m] = int(val)
        else:
            colresult[m] = val

    colresult["n"] = column_props[col]["notnulls"]

    percentiles = [0.1, 1, 10, 25, 50, 75, 90, 99, 99.9]
    colresult["percentiles"] = {
        perc: np.nanpercentile(series, perc) for perc in percentiles
    }
    colresult["median"] = colresult["percentiles"][50]
    colresult["iqr"] = (
        colresult["percentiles"][75] - colresult["percentiles"][25]
    )

    # Compute the t-digest.
    logger.debug("column_summary - {} - creating TDigest...".format(col))
    digest = TDigest(delta)
    digest.batch_update(data)

    logger.debug("column_summary - {} - testing log trans...".format(col))
    try:
        colresult["logtrans"] = bool(_test_logtrans(digest))
    except Exception as e:
        # Hard to pinpoint problems with the logtrans TDigest.
        logger.warning(
            "test_logtrans has failed for column `{}`: {}".format(col, e)
        )
        colresult["logtrans"] = False

    if colresult["logtrans"]:
        logdigest = TDigest()
        for c in digest.C.values():
            logdigest.update(np.log(c.mean), c.count)
        colresult["logtrans_mean"] = _tdigest_mean(logdigest)
        colresult["logtrans_std"] = _tdigest_std(logdigest)
        colresult["logtrans_IQR"] = logdigest.percentile(
            75
        ) - logdigest.percentile(25)

    logger.debug(
        "column_summary - {} - should {}be log-transformed".format(
            col, "NOT " if not colresult["logtrans"] else ""
        )
    )

    # Compress and store the t-digest.
    digest.delta = delta
    digest.compress()
    colresult["tdigest"] = [(c.mean, c.count) for c in digest.C.values()]

    # Compute histogram
    logger.debug("column_summary - {} - computing histogram...".format(col))

    if column_props[col]["is_categorical"]:
        # Compute frequency table and store as histogram
        counts, edges = _compute_histogram_from_frequencies(data)
    else:
        if colresult["logtrans"]:
            counts, log_edges = np.histogram(
                np.log10(data), density=False, bins="fd"
            )
            edges = 10 ** log_edges
        else:
            counts, edges = np.histogram(data, density=False, bins="fd")

    colresult["histogram"] = {
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
    }

    # Compute KDE
    logger.debug("column_summary - {} - computing KDE...".format(col))
    bw = _bw_scott(colresult, colresult["n"], colresult["logtrans"], 1)

    logger.debug("column_summary - {} - KDE bw: {:.4g}".format(col, bw))

    if column_props[col]["is_categorical"]:
        kde_x, kde_y = np.zeros(1), np.zeros(1)
    else:
        coord_range = colresult["min"], colresult["max"]
        kde_x, kde_y = _compute_smoothed_histogram(
            data, bw, coord_range, logtrans=colresult["logtrans"]
        )

    colresult["kde"] = {"x": kde_x.tolist(), "y": kde_y.tolist()}

    return {col: colresult, "_columns": [col]}


def _compute_histogram_from_frequencies(series):
    """Compute histogram from frequencies

    This method uses the frequencies dict to produce a histogram data structure
    with emtpy bins where the difference between the category values is larger
    than 1

    Parameters
    ----------
    series : pd.Series
        Categorical column.a

    Returns
    -------
    counts, edges:
        Histogram bin edges and counts in each bin.
    """
    freqs = _compute_frequencies(series)
    categories = sorted(freqs.keys())
    diffs = list(np.diff(categories)) + [1]
    edges = [categories[0] - 0.5]
    counts = []
    for cat, diff in zip(categories, diffs):
        if diff <= 1:
            edges.append(cat + diff / 2.)
            counts.append(freqs[cat])
        else:
            edges += [cat + 0.5, cat + diff - 0.5]
            counts += [freqs[cat], 0]

    return np.array(counts), np.array(edges)


def _compute_frequencies(series):
    """Helper to compute frequencies of a categorical column

    Parameters
    ----------
    series : pd.Series
        Categorical column.a

    Returns
    -------
    dict:
        Dictionary from category name to count.
    """
    freqs = series.value_counts()
    if freqs.index.dtype == np.int64:
        categories = [int(index) for index in freqs.index]
    elif freqs.index.dtype == np.float64:
        categories = [float(index) for index in freqs.index]
    else:
        categories = freqs.index
    return dict(zip(categories, freqs.values.tolist()))


@timeit
def frequencies(series, column_props):
    """Compute frequencies for categorical columns.

    Parameters
    ----------
    series : pd.Series
        Categorical column.
    column_props : dict
        Dictionary as returned by `column_properties`

    Returns
    -------
    TODO
    """
    name = series.name

    if column_props[name]["is_categorical"]:
        logger.debug("frequencies - " + series.name)
        freqs = _compute_frequencies(series)
        return {name: freqs, "_columns": [name]}
    else:
        return None


@timeit
def outliers(series, column_summ):
    """Count outliers for numeric columns.

    Parameters
    ----------
    series : pd.Series
        Numeric column.
    column_summ : TODO
        TODO

    Returns
    -------
    TODO
    """
    name = series.name
    if column_summ is None:
        # Not a numeric column.
        return None
    else:
        column_summ = column_summ[name]

    Q1, Q3 = [column_summ["percentiles"][p] for p in [25, 75]]
    IQR = Q3 - Q1
    # Mild outlier limits.
    lom = Q1 - 1.5 * IQR
    him = Q3 + 1.5 * IQR
    # Extreme outlier limits.
    lox = Q1 - 3.0 * IQR
    hix = Q3 + 3.0 * IQR

    nn = series.dropna()

    Nmildlo = len(nn[(nn < lom) & (nn > lox)].index)
    Nmildhi = len(nn[(nn > him) & (nn < hix)].index)
    Nextrlo = len(nn[nn < lox].index)
    Nextrhi = len(nn[nn > hix].index)

    return {
        name: {"mild": [Nmildlo, Nmildhi], "extreme": [Nextrlo, Nextrhi]},
        "_columns": [name],
    }


@timeit
def correlation(df, column_props):
    """Compute correlation table between non-ID numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame.
    column_props : TODO
        TODO

    Returns
    -------
    dict
        Dictionary containing correlation coefficients.
    """

    cols = [
        col
        for col in df.columns
        if (column_props[col]["numeric"] and not column_props[col]["is_ID"])
    ]

    numdf = df[cols]
    pcorr = numdf.corr(method="pearson", min_periods=5)
    scorr = numdf.corr(method="spearman", min_periods=5)

    report = {}
    report["_columns"] = list(numdf.columns)
    report["pearson"] = np.array(pcorr).tolist()
    report["spearman"] = np.array(scorr).tolist()

    report["order"] = hierarchical_ordering_indices(
        numdf.columns, scorr.values
    )

    return report


def _compute_smoothed_histogram(
    values, bandwidth, coord_range, logtrans=False
):
    """Approximate 1-D density estimation.

    Estimate 1-D probability densities at evenly-spaced grid points,
    for specified data. This method is based on creating a 1-D histogram of
    data points quantised with respect to evenly-spaced grid points.
    Probability densities are then estimated at the grid points by convolving
    the obtained histogram with a Gaussian kernel.

    Parameters
    ----------
    values : np.array (N,)
        A vector containing the data for which to perform density estimation.
        Successive data points are indexed by the first axis in the array.
    bandwidth : float
        The desired KDE bandwidth. (When log-transformation
        of data is desired, bandwidth should be specified in log-space.)
    coord_range: (2,)
        Minimum and maximum values of coordinate on which to evaluate the
        smoothed histogram.
    logtrans : boolean
        Whether or not to log-transform the data before performing density
        estimation.

    Returns
    -------
    np.array (M-1,)
    An array of estimated probability densities at specified grid points.
    """
    if logtrans:
        ber = [np.log10(extreme) for extreme in coord_range]
        bin_edges = np.logspace(*ber, num=DENSITY_N + 1)
        bin_edge_range = ber[1] - ber[0]
    else:
        bin_edges = np.linspace(*coord_range, num=DENSITY_N + 1)
        bin_edge_range = coord_range[1] - coord_range[0]

    if values.size < 2:
        # Return zeros if there are too few points to do anything useful.
        return bin_edges[:-1], np.zeros(bin_edges.shape[0] - 1)

    # Bin the values
    H = np.histogram(values, bin_edges)[0]

    relative_bw = bandwidth / bin_edge_range
    K = _compute_gaussian_kernel(H.shape, relative_bw)

    pdf = signal.fftconvolve(H, K, mode="same")

    # Return lower edges of bins and normalized pdf
    return bin_edges[:-1], pdf / np.trapz(pdf, bin_edges[:-1])


def _compute_smoothed_histogram2d(
    values, bandwidth, coord_ranges, logtrans=False
):
    """Approximate 2-D density estimation.

    Estimate 2-D probability densities at evenly-spaced grid points,
    for specified data. This method is based on creating a 2-D histogram of
    data points quantised with respect to evenly-spaced grid points.
    Probability densities are then estimated at the grid points by convolving
    the obtained histogram with a Gaussian kernel.

    Parameters
    ----------
    values : np.array (N,2)
        A 2-D array containing the data for which to perform density
        estimation. Successive data points are indexed by the first axis in the
        array. The second axis indexes x and y coordinates of data points
        (values[:,0] and values[:,1] respectively).
    bandwidth : array-like (2,)
        The desired KDE bandwidths for x and y axes. (When log-transformation
        of data is desired, bandwidths should be specified in log-space.)
    coord_range: (2,2)
        Minimum and maximum values of coordinates on which to evaluate the
        smoothed histogram.
    logtrans : array-like (2,)
        A 2-element boolean array specifying whether or not to log-transform
        the x or y coordinates of the data before performing density
        estimation.

    Returns
    -------
    np.array (M-1, M-1)
        An array of estimated probability densities at specified grid points.
    """
    bin_edges = []
    bedge_range = []
    for minmax, lt in zip(coord_ranges, logtrans):
        if lt:
            ber = [np.log10(extreme) for extreme in minmax]
            bin_edges.append(np.logspace(*ber, num=DENSITY_N + 1))
            bedge_range.append(ber[1] - ber[0])
        else:
            bin_edges.append(np.linspace(*minmax, num=DENSITY_N + 1))
            bedge_range.append(minmax[1] - minmax[0])

    # Bin the observations
    H = np.histogram2d(values[:, 0], values[:, 1], bins=bin_edges)[0]

    relative_bw = [bw / berange for bw, berange in zip(bandwidth, bedge_range)]
    K = _compute_gaussian_kernel(H.shape, relative_bw)

    pdf = signal.fftconvolve(H.T, K, mode="same")

    # Normalize pdf
    bin_centers = [edges[:-1] + np.diff(edges) / 2. for edges in bin_edges]
    pdf /= np.trapz(np.trapz(pdf, bin_centers[1]), bin_centers[0])

    # Return lower bin edges and density
    return bin_edges[0][:-1], bin_edges[1][:-1], pdf


def _compute_gaussian_kernel(histogram_shape, relative_bw):
    """Compute a gaussian kernel double the size of the histogram matrix"""
    if len(histogram_shape) == 2:
        kernel_shape = [2 * n for n in histogram_shape]
        # Create a scaled grid in which the kernel is symmetric to avoid matrix
        # inversion problems when the bandwiths are very different
        bw_ratio = relative_bw[0] / relative_bw[1]
        bw = relative_bw[0]
        X, Y = np.mgrid[
            -bw_ratio : bw_ratio : kernel_shape[0] * 1j,
            -1 : 1 : kernel_shape[1] * 1j,
        ]
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        Cov = np.array(((bw, 0), (0, bw))) ** 2
        K = stats.multivariate_normal.pdf(grid_points, mean=(0, 0), cov=Cov)

        return K.reshape(kernel_shape)
    else:
        grid = np.mgrid[-1 : 1 : histogram_shape[0] * 2j]
        return stats.norm.pdf(grid, loc=0, scale=relative_bw)


def _bw_scott(column_summ, N, logtrans, d):
    """Scott's rule of thumb for KDE kernel bandwidth.

    Parameters
    ----------
    column_summ : dict
        Dictionary as returned by `column_summary`.
    N : int
        Number of elements in the series for which the KDE is to be
        evaluated.
    logtrans : bool
        Whether the series is assumed to be 'exponential' (True) or
        'linear' (False). An 'exponential' series (representing, e.g.
        income) is log-transformed before the KDE. The bandwidth
        therefore needs to be estimated for the log transformed series.
    d : int
        Dimension of the KDE.

    Returns
    -------
    float
        Estimate of the kernel bandwidth for the KDE.
    """
    if N == 0:
        return 0

    norm = 1.349  # norm.ppf(0.75) - norm.ppf(0.25)
    if logtrans:
        std, IQR = column_summ["logtrans_std"], column_summ["logtrans_IQR"]
        factor = 2
    else:
        std, IQR = column_summ["std"], column_summ["iqr"]
        factor = 1.4

    if IQR > 0:
        iqr_estimate = min(IQR / norm, std)
    elif std > 0:
        iqr_estimate = std
    else:
        iqr_estimate = 1.0

    bandwidth = 1.06 * iqr_estimate * N ** (-1. / (4. + d))

    return bandwidth / factor


@timeit
def pairdensity(df, column_props, column_summ, freq, log_transform=True):
    """Compute a variable pair heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the columns for which the pair density is
        computed.
    column_props : dict
        Column properties dictionary with at least col1 and col2, as
        returned by `column_properties`.
    column_summ : dict
        Column summary dictionary with at least col1 and col2, as
        returned by `column_summary`.
    freq : dict
        Frequencies dictionary with at least col1 and col2.
    log_transform : bool
        Whether to compute the KDE in log-space when needed.

    Returns
    -------
    TODO
    """
    col1, col2 = df.columns

    # Test that both columns have valid entries and are either
    # categorical or numeric, returning None if not.
    column_props = {col: column_props[col][col] for col in [col1, col2]}
    for col in [col1, col2]:
        if (
            not (
                column_props[col]["is_categorical"]
                or column_props[col]["numeric"]
            )
            or column_props[col]["notnulls"] == 0
        ):
            return None

    report = {"_columns": [col1, col2], col1: {}}

    log_string = "pairdensity - {} - {}".format(col1, col2)
    logger.debug("{}".format(log_string))

    data = df.dropna()
    N = len(data.index)

    coord_ranges, scales, categories = [], [], []
    bandwidths = [None, None]
    for col in [col1, col2]:
        if column_props[col]["is_categorical"]:
            scales.append("category")
            coord_ranges.append(None)
            categories.append(sorted(list(freq[col][col].keys())))
        else:
            scales.append(
                "log" if column_summ[col][col]["logtrans"] else "linear"
            )
            coord_ranges.append(
                [column_summ[col][col][extreme] for extreme in ["min", "max"]]
            )
            categories.append(None)

    Ncat = np.sum([scale == "category" for scale in scales])

    if N == 0:
        logger.warning("{}: No valid pairs found!".format(log_string))

    if Ncat == 0:
        # 2D pair density is not useful with very few observations
        if N > 3:
            logtrans = [scale == "log" for scale in scales]

            bandwidths = [
                _bw_scott(column_summ[col][col], N, lt, 2 - Ncat)
                for col, lt in zip([col1, col2], logtrans)
            ]

            x, y, density = _compute_smoothed_histogram2d(
                np.array(data), bandwidths, coord_ranges, logtrans=logtrans
            )

            x, y = x.tolist(), y.tolist()
        else:
            x, y = coord_ranges
            density = np.zeros((2, 2))

    elif Ncat == 1:
        # Split into categories and do a univariate KDE on each.
        if column_props[col1]["is_categorical"]:
            cats = categories[0]
            coord_range = coord_ranges[1]
            catcol, numcol, numcolsum = col1, col2, column_summ[col2][col2]
            logtrans = scales[1] == "log"
        else:
            cats = categories[1]
            coord_range = coord_ranges[0]
            catcol, numcol, numcolsum = col2, col1, column_summ[col1][col1]
            logtrans = scales[0] == "log"

        density = []
        for cat in cats:
            # Filter data for this category.
            datacat = data[data[catcol] == cat][numcol]
            Nincat = datacat.count()

            # Recompute the bandwidth because the number of pairs in
            # this category might be lower than the total number of
            # pairs.
            num_bw = _bw_scott(numcolsum, Nincat, logtrans, 1)
            grid, catdensity = _compute_smoothed_histogram(
                datacat, num_bw, coord_range, logtrans=logtrans
            )

            # Remove normalisation to normalise it later to the total
            # number of pairs.
            density.append(catdensity * Nincat)

        density = np.array(density) / N

        if column_props[col1]["is_categorical"]:
            density = density.T
            x, y = cats, grid.tolist()
        else:
            x, y = grid.tolist(), cats

    elif Ncat == 2:
        if N > 0:
            # Crosstab frequencies.
            dfcs = (
                pd.crosstab(data[col2], data[col1])
                .sort_index(axis=0)
                .sort_index(axis=1)
            )

            x = [str(column) for column in dfcs.columns]
            if "" in x:
                x[x.index("")] = " Null"

            y = [str(index) for index in dfcs.index]
            if "" in y:
                y[y.index("")] = " Null"

            density = dfcs.get_values()
        else:
            x, y = categories
            density = np.zeros((len(x), len(y)))

    report[col1][col2] = {
        "density": density.tolist(),
        "axes": {col1: x, col2: y},
        "bw": bandwidths,
        "scales": scales,
    }

    return report
