"""Plotting utils, mostly adapted from seaborn for use with TDigests."""
import numpy as np
from scipy import stats
from six import string_types

import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as hierarchy


def _kde_support(data, bw, gridsize, cut, clip):
    """Establish support for a kernel density estimate."""
    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])
    return np.linspace(support_min, support_max, gridsize)


def _scipy_univariate_kde(data, bw, gridsize, cut, clip):
    """Compute a univariate kernel density estimate using scipy."""
    kde = stats.gaussian_kde(data, bw_method=bw)
    if isinstance(bw, string_types):
        bw = "scotts" if bw == "scott" else bw
        bw = getattr(kde, "%s_factor" % bw)() * np.std(data)
    grid = _kde_support(data, bw, gridsize, cut, clip)
    y = kde(grid)
    return grid, y


def _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using scipy."""
    data = np.c_[x, y]
    kde = stats.gaussian_kde(data.T)
    data_std = data.std(axis=0, ddof=1)
    if isinstance(bw, string_types):
        bw = "scotts" if bw == "scott" else bw
        bw_x = getattr(kde, "%s_factor" % bw)() * data_std[0]
        bw_y = getattr(kde, "%s_factor" % bw)() * data_std[1]
    elif np.isscalar(bw):
        bw_x, bw_y = bw, bw
    else:
        msg = (
            "Cannot specify a different bandwidth for each dimension "
            "with the scipy backend. You should install statsmodels."
        )
        raise ValueError(msg)
    x_support = _kde_support(data[:, 0], bw_x, gridsize, cut, clip[0])
    y_support = _kde_support(data[:, 1], bw_y, gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


def axis_ticklabels_overlap(labels):
    """Return a boolean for whether the list of ticklabels have overlaps.

    Parameters
    ----------
    labels : list of ticklabels

    Returns
    -------
    overlap : boolean
        True if any of the labels overlap.
    """
    if not labels:
        return False
    try:
        bboxes = [l.get_window_extent() for l in labels]
        overlaps = [b.count_overlaps(bboxes) for b in bboxes]
        return max(overlaps) > 1
    except RuntimeError:
        # Issue on macosx backend rasies an error in the above code
        return False


def hierarchical_ordering_indices(columns, correlation_matrix):
    """Return array with hierarchical cluster ordering of columns

    Parameters
    ----------
    columns: iterable of str
        Names of columns.
    correlation_matrix: np.ndarray
        Matrix of correlation coefficients between columns.

    Returns
    -------
    indices: iterable of int
        Indices with order of columns
    """
    if len(columns) > 2:
        pairwise_dists = distance.pdist(
            np.where(np.isnan(correlation_matrix), 0, correlation_matrix),
            metric="euclidean",
        )
        linkage = hierarchy.linkage(pairwise_dists, method="average")
        dendogram = hierarchy.dendrogram(
            linkage, no_plot=True, color_threshold=-np.inf
        )
        idx = dendogram["leaves"]
    else:
        idx = list(range(len(columns)))

    return idx


def hierarchical_ordering(columns, correlation_matrix):
    """Reorder matrix by hierarchical clustering of columns

    Parameters
    ----------
    columns: iterable of str
        Names of columns.
    correlation_matrix: np.ndarray
        Matrix of correlation coefficients between columns.

    Returns
    ------
    columns: iterable of str
        Reorderd names of columns.
    correlation_matrix: np.ndarray
        Reordered matrix of correlation coefficients between columns.
    """
    if len(columns) > 2:
        idx = hierarchical_ordering_indices(columns, correlation_matrix)
        correlation_matrix = correlation_matrix[idx, :][:, idx]
        columns = [columns[i] for i in idx]

    return columns, correlation_matrix
