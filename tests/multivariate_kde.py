import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def _KDEMultivariate(
    values, bandwidth=None, grid_points=None, grid_shape=None, logtrans=None
):
    """Multivariate kernel density estimation."""

    if values.shape[0] < 3:
        # Return zeroes if there are too few points to do anything
        # useful.
        return np.zeros(grid_shape)

    for i, lt in enumerate(logtrans):
        if lt:
            values[:, i] = np.log10(values[:, i])
            grid_points[i] = np.log10(grid_points[i])

    kernel = sm.nonparametric.KDEMultivariate(
        data=values, var_type="cc", bw=bandwidth
    )

    pdf = np.reshape(kernel.pdf(grid_points), grid_shape)

    return pdf


def plot_pd_difference(hist_pd, kde_pd, filename):
    max_dev = np.max(np.abs(kde_pd - hist_pd))
    max_dev_s = "Max dev: {:.3g}".format(max_dev)

    I = np.sum((hist_pd > 1E-3) + (kde_pd > 1E-3)) / 2.0
    mean_dev = np.sum(np.abs(kde_pd - hist_pd)) / I
    mean_dev_s = "Mean dev: {:.3g}".format(mean_dev)

    corr = 1 - np.corrcoef(kde_pd, hist_pd)[0][1]
    corr_s = "1 - Corr: {:.3g}".format(corr)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    ax1.pcolormesh(hist_pd, cmap="viridis")
    ax1.set_aspect("equal")
    ax1.set_title("Smoothed Hist")

    ax2.pcolormesh(kde_pd, cmap="viridis")
    ax2.set_aspect("equal")
    ax2.set_title("2D KDE")

    diff = hist_pd - kde_pd

    diff_img = ax3.pcolormesh(diff * 100., cmap="RdBu", vmin=-10, vmax=10)
    ax3.set_aspect("equal")
    ax3.set_title("Difference")

    f.tight_layout()

    f.colorbar(diff_img, ax=[ax1, ax2, ax3], label="% difference")
    f.text(0.9, 0.5, "\n".join([max_dev_s, mean_dev_s, corr_s]), va="center")

    f.savefig(filename)
    plt.close(f)


def _normalise_range(X):
    return X / np.max(X)


def compute_deviation_with_kde(df, pd, filename):
    """ Compute mean deviation of smoothed histogram with respect to KDE """
    columns = pd["_columns"]
    pd = pd[columns[0]][columns[1]]
    bw = pd["bw"]
    logtrans = [scale == "log" for scale in pd["scales"]]
    x = pd["axes"][columns[0]]
    y = pd["axes"][columns[1]]
    X, Y = np.meshgrid(x, y)
    grid_shape = X.shape
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    kde_pd = _KDEMultivariate(
        np.array(df.dropna()),
        bandwidth=bw,
        grid_points=grid_points,
        grid_shape=grid_shape,
        logtrans=logtrans,
    )
    hist_pd = np.array(pd["density"])

    # hist_pd[50] = hist_pd[50] + np.mean(hist_pd) * 0.1

    kde_pd = _normalise_range(kde_pd)
    hist_pd = _normalise_range(hist_pd)

    I = np.sum((hist_pd > 1E-3) + (kde_pd > 1E-3)) / 2.0
    mean_dev = np.sum(np.abs(kde_pd - hist_pd)) / I

    if mean_dev > 0.01:
        plot_pd_difference(hist_pd, kde_pd, filename)

    return mean_dev
