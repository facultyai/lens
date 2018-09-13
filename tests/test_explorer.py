import inspect
import os

import pandas as pd
import matplotlib.pyplot as plt

import pytest

import lens
from lens.explorer import Explorer

dirname = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)


@pytest.fixture(scope="module")
def artworks_df():
    df = pd.read_csv(os.path.join(dirname, "data/test-artworks.csv"))
    return df


@pytest.fixture(scope="module")
def artworks_summary(artworks_df):
    summary = lens.summarise(artworks_df)
    return summary


def test_distribution_plot(artworks_df, artworks_summary):
    def mock_render(fig):
        # check that this draws a histogram
        assert len(fig.axes[0].patches) > 0

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.distribution_plot("Height (cm)")


def test_distribution_plot_bins(artworks_df, artworks_summary):
    Nbins = 13

    def mock_render(fig):
        # check that this draws a histogram with Nbins bars
        assert len(fig.axes[0].patches) == Nbins

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.distribution_plot("Height (cm)", bins=Nbins)


def test_cdf_plot(artworks_df, artworks_summary):
    column = "Height (cm)"
    plt.cla()

    def mock_render(fig):
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        line = ax.lines[0]

        tdigest = artworks_summary.tdigest(column)
        xs = [tdigest.percentile(p) for p in [0, 100]]

        assert line.get_xdata()[0] == xs[0]
        assert line.get_xdata()[-1] == xs[-1]
        assert line.get_ydata()[0] == 0
        assert line.get_ydata()[-1] == 100

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.cdf_plot("Height (cm)")


def test_cdf_plot_log_transformed(artworks_df, artworks_summary):
    plt.cla()

    def mock_render(fig):
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        assert ax.get_xaxis().get_scale() == "log"

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.cdf_plot("Width (cm)")


def test_cdf_plot_non_numeric(artworks_summary):
    def mock_render(fig):
        assert False

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    with pytest.raises(ValueError):
        explorer.cdf_plot("Nationality")


def test_pairwise_density_plot(artworks_df, artworks_summary):
    plt.cla()

    def mock_render(fig):
        #  currently pairwise_density_plot returns a plotly figure
        assert len(fig["data"]) == 1
        data = fig["data"][0]
        assert data["type"] == "heatmap"

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.pairwise_density_plot("Width (cm)", "Height (cm)")


def test_pairwise_density_plot_one_categorical(artworks_df, artworks_summary):
    plt.cla()

    def mock_render(fig):
        #  currently pairwise_density_plot returns a plotly figure
        assert len(fig["data"]) == 1
        data = fig["data"][0]
        assert data["type"] == "heatmap"

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.pairwise_density_plot("Nationality", "Height (cm)")
    explorer.pairwise_density_plot("Height (cm)", "Nationality")


def test_pairwise_density_plot_both_categorical(artworks_df, artworks_summary):
    plt.cla()

    def mock_render(fig):
        #  currently pairwise_density_plot returns a plotly figure
        assert len(fig["data"]) == 1
        data = fig["data"][0]
        assert data["type"] == "heatmap"

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.pairwise_density_plot("Nationality", "Gender")


def test_pairwise_density_plot_not_numeric(artworks_df, artworks_summary):
    plt.cla()

    def mock_render(fig):
        assert False

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    with pytest.raises(ValueError):
        explorer.pairwise_density_plot("Diameter (cm)", "Nationality")


def test_correlation_plot(artworks_df, artworks_summary):
    plt.cla()

    def mock_render(fig):
        assert len(fig["data"]) == 1
        data = fig["data"][0]
        assert data["type"] == "heatmap"
        expected_columns = {"Height (cm)", "Width (cm)", "Depth (cm)"}
        assert set(data["y"]) == set(data["x"]) == expected_columns

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.correlation_plot()


def test_correlation_plot_annotations(artworks_df, artworks_summary):
    plt.cla()

    def mock_render(fig):
        assert len(fig["data"]) == 1
        corr = [item for row in fig["data"][0]["z"] for item in row]
        labels = [l["text"] for l in fig["layout"]["annotations"]]
        for c, l in zip(corr, labels):
            assert "{:.2g}".format(c) == l

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.correlation_plot(
        include=["Height (cm)", "Width (cm)", "Depth (cm)"]
    )
    explorer.correlation_plot(include=["Height (cm)", "Width (cm)"])


def test_correlation_plot_include(artworks_df, artworks_summary):
    plt.cla()

    def mock_render(fig):
        assert len(fig["data"]) == 1
        data = fig["data"][0]
        assert data["type"] == "heatmap"
        assert set(data["y"]) == set(data["x"]) == {"Height (cm)"}

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.correlation_plot(include=["Height (cm)"])


def test_correlation_plot_exclude(artworks_df, artworks_summary):
    plt.cla()

    def mock_render(fig):
        assert len(fig["data"]) == 1
        data = fig["data"][0]
        assert data["type"] == "heatmap"
        expected_columns = {"Height (cm)", "Depth (cm)"}
        assert set(data["y"]) == set(data["x"]) == expected_columns

    explorer = Explorer(artworks_summary, plot_renderer=mock_render)
    explorer.correlation_plot(exclude=["Width (cm)"])
