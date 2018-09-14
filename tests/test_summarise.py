import json
import inspect
import os
import itertools
import datetime

import numpy as np
import pandas as pd

import pytest

from lens import summarise, metrics, __version__
from lens.summarise import EmptyDataFrameError, NumpyEncoder
from lens.dask_graph import _join_dask_results
from lens.metrics import CAT_FRAC_THRESHOLD
from fixtures import df, report  # noqa

from multivariate_kde import compute_deviation_with_kde

dirname = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)

test_results_dir = dirname + "/test_results"

if not os.path.exists(test_results_dir):
    os.mkdir(test_results_dir)


def test_dask_row_count(df):
    rc_report = metrics.row_count(df)
    assert rc_report["total"] == len(df)
    assert rc_report["unique"] == len(df.drop_duplicates().index)

    # test serialization
    json.dumps({"row_count": rc_report}, cls=NumpyEncoder)


def test_zero_rows_dataframe():
    columns = sorted(["a", "b", "c", "d"])
    df = pd.DataFrame(columns=columns)
    report = summarise(df)._report
    assert sorted(report["_columns"]) == columns
    for column in columns:
        props = report["column_properties"][column]
        assert props["nulls"] == 0
        assert props["notnulls"] == 0
        assert props["unique"] == 0


def test_one_row_dataframe():
    items = [
        ("a", [1]),
        ("b", [-0.5]),
        ("c", ["hello"]),
        ("d", [datetime.datetime.now()]),
    ]
    columns = sorted([item[0] for item in items])
    df = pd.DataFrame.from_dict(dict(items))
    report = summarise(df)._report
    assert sorted(report["_columns"]) == columns
    column_properties = report["column_properties"]
    for column in columns:
        props = column_properties[column]
        assert props["nulls"] == 0
        assert props["notnulls"] == 1
        assert props["unique"] == 1
    assert column_properties["a"]["dtype"] == "int64"
    assert column_properties["b"]["dtype"] == "float64"
    assert column_properties["c"]["dtype"] == "object"
    assert column_properties["d"]["dtype"] == "datetime64[ns]"
    column_summary = report["column_summary"]
    assert column_summary["a"]["max"] == 1
    assert column_summary["a"]["min"] == 1
    assert column_summary["a"]["mean"] == 1.0
    assert column_summary["a"]["median"] == 1.0
    assert column_summary["a"]["iqr"] == 0.0

    assert column_summary["b"]["max"] == -0.5
    assert column_summary["b"]["min"] == -0.5
    assert column_summary["b"]["median"] == -0.5
    assert column_summary["b"]["mean"] == -0.5


@pytest.fixture(scope="module")
def column_properties(df):
    cols = df.columns
    cps = {col: metrics.column_properties(df[col]) for col in cols}

    return cps


@pytest.fixture(scope="module")
def column_summary(df, column_properties):
    cols = df.columns
    cs = {
        col: metrics.column_summary(df[col], column_properties[col])
        for col in cols
    }
    return cs


def test_dask_column_properties(column_properties):
    # Only worth checking that we determine categorical columns
    # correctly if there are enough rows in the dataframe.
    # There are 13 distinct categories.
    categorical13_props = column_properties["categorical13"]["categorical13"]
    row_threshold = 2 * 13.0 / CAT_FRAC_THRESHOLD
    if categorical13_props["notnulls"] > row_threshold:
        assert categorical13_props["is_categorical"]

    # test serialization
    joined = _join_dask_results(column_properties.values()).compute()
    json.dumps({"column_summary": joined}, cls=NumpyEncoder)


def test_dask_column_summary(df, column_summary):
    for col in df.columns:
        series = df[col]
        cs_report = column_summary[col]

        if cs_report is None or series.isnull().sum() == len(df.index):
            continue

        else:
            cs_report = cs_report[col]

        # Test that only lognormal is set to log transform
        # Only run this test if the column has enough valid
        # values
        if len(df.index) >= 50:
            if col == "lognormal":
                assert cs_report["logtrans"]
            else:
                assert not cs_report["logtrans"]

        _percs = list(cs_report["percentiles"].keys())
        _percs.sort()
        cs_report_perc = [cs_report["percentiles"][p] for p in _percs]
        exact_perc = np.nanpercentile(series, _percs)
        np.testing.assert_allclose(
            cs_report_perc, exact_perc, rtol=1e-3, atol=1e-3
        )

        exact_meanminmax = [
            np.nanmean(series.get_values()),
            np.nanmin(series.get_values()),
            np.nanmax(series.get_values()),
        ]
        rep_meanminmax = [cs_report[x] for x in ["mean", "min", "max"]]
        np.testing.assert_allclose(
            exact_meanminmax, rep_meanminmax, rtol=1e-3, atol=0.01
        )

        # test histogram
        histogram = cs_report["histogram"]

        assert np.sum(histogram["counts"]) == series.notnull().sum()
        if cs_report["n"] > 1 and not np.all(np.mod(series.dropna(), 1) == 0):
            # Bin edges for single-count histograms are not relevant, and
            # integer-only histograms not bounded by extremes in distribution
            assert np.allclose(histogram["bin_edges"][0], series.min())
            assert np.allclose(histogram["bin_edges"][-1], series.max())

        if col == "categoricalint":
            # Check that bins are set correctly for integers
            # we are removing the twos so there should be at least one empty
            # bin in the histogram
            n_unique = series.dropna().unique().size
            assert len(histogram["counts"]) >= n_unique
            assert len(histogram["bin_edges"]) == len(histogram["counts"]) + 1

            # Check that the bin that contains 2 is set to 0
            idx = np.where(np.array(histogram["bin_edges"]) < 2)[0][-1]
            assert histogram["counts"][idx] == 0

            assert np.allclose(
                histogram["bin_edges"][0], series.dropna().min() - 0.5
            )
            assert np.allclose(
                histogram["bin_edges"][-1], series.dropna().max() + 0.5
            )

        # test kde
        kde = cs_report["kde"]

        assert np.all(~np.isnan(kde["x"]))
        assert np.all(~np.isnan(kde["y"]))

        if "categorical" not in col and np.sum(kde["y"]) > 0:
            assert np.allclose(np.trapz(kde["y"], kde["x"]), 1)

        if col == "normal":
            mean = cs_report["mean"]
            kde_max = kde["x"][np.argmax(kde["y"])]
            assert np.allclose(kde_max, mean, atol=5, rtol=0.1)

    # test serialization
    joined = _join_dask_results(column_summary.values()).compute()
    json.dumps({"column_summary": joined}, cls=NumpyEncoder)


def test_dask_outliers(df, column_summary):
    reps = []
    for col in df.columns:
        reps.append(metrics.outliers(df[col], column_summary[col]))

    # test serialization
    joined = _join_dask_results(reps).compute()
    json.dumps({"outliers": joined}, cls=NumpyEncoder)


@pytest.fixture(scope="module")
def frequencies(df, column_properties):
    return {
        col: metrics.frequencies(df[col], column_properties[col])
        for col in df.columns
    }


def test_dask_frequencies(df, frequencies):
    for col in frequencies.keys():
        freq_report = frequencies[col]
        if freq_report is None:
            continue
        else:
            freq_report = freq_report[col]

        freqs = df[col].value_counts().to_dict()

        for k in freqs.keys():
            assert freqs[k] == freq_report[k]

    # test serialization
    joined = _join_dask_results(frequencies.values()).compute()
    json.dumps({"freqs": joined}, cls=NumpyEncoder)


def test_dask_correlation(df, column_properties):
    cp = _join_dask_results(column_properties.values()).compute()
    rep = metrics.correlation(df, cp)
    cols = rep["_columns"]
    sp = np.array(rep["spearman"])
    order = rep["order"]

    assert len(order) == len(cols)
    assert sp.shape[0] == len(cols)
    assert sp.shape[1] == len(cols)

    # test serialization
    json.dumps({"correlation": rep}, cls=NumpyEncoder)


def test_dask_pairdensity(df, column_properties, column_summary, frequencies):
    pds = []
    for col1, col2 in itertools.combinations(df.columns, 2):
        cp = {k: column_properties[k] for k in [col1, col2]}
        cs = {k: column_summary[k] for k in [col1, col2]}
        fr = {k: frequencies[k] for k in [col1, col2]}
        pd = metrics.pairdensity(df[[col1, col2]], cp, cs, fr)
        if pd is not None:
            if should_pair_density_norm_be_finite(df[[col1, col2]], cp):
                if (
                    not cp[col1][col1]["is_categorical"]
                    and not cp[col2][col2]["is_categorical"]
                    and "poisson" not in col1
                    and "poisson" not in col2
                ):
                    filename = "{}/{}_{}_{}_pd_diff.png".format(
                        test_results_dir, len(df.index), col1, col2
                    )
                    mean_dev = compute_deviation_with_kde(
                        df[[col1, col2]], pd, filename
                    )
                    assert mean_dev < 0.02
                assert (
                    np.sum(pd[col1][col2]["density"]) > 0
                ), "Failed on columns {} - {}".format(col1, col2)

        pds.append(pd)

    joined = _join_dask_results(pds).compute()

    # test serialization
    json.dumps({"pairdensity": joined}, cls=NumpyEncoder)


def should_pair_density_norm_be_finite(df, column_properties):
    col1, col2 = df.columns
    valid_rows = df.dropna().index
    is_col1_categorical = column_properties[col1][col1]["is_categorical"]
    is_col2_categorical = column_properties[col2][col2]["is_categorical"]
    if is_col1_categorical and is_col2_categorical:
        return len(valid_rows) >= 1
    elif is_col1_categorical:
        n_distinct = column_properties[col1][col1]["unique"]
        return len(valid_rows) >= (n_distinct * 2)
    elif is_col2_categorical:
        n_distinct = column_properties[col2][col2]["unique"]
        return len(valid_rows) >= (n_distinct * 2)
    else:
        return len(valid_rows) >= 3


def serialize_full_report(dreport, fname=None):
    # test that it can be serialized as json
    try:
        if fname is None:
            json.dumps(dreport, cls=NumpyEncoder)
        else:
            with open(fname, "w") as f:
                json.dump(dreport, f, indent=2)
    except TypeError:
        # Nail down which metric is failing
        for k in dreport.keys():
            try:
                json.dumps({k: dreport[k]}, cls=NumpyEncoder)
            except TypeError as e:
                raise TypeError(
                    "Metric {} is not JSON serializable: {}".format(k, e)
                )


def test_dask_compute_graph_default(report):
    fname = "{}/test_results/report_test_data.json".format(dirname)

    serialize_full_report(report, fname=fname)


@pytest.mark.parametrize(
    "scheduler,num_workers,pairdensities",
    [
        ("sync", None, True),
        ("multiprocessing", 2, True),
        ("threading", None, True),
        ("multiprocessing", 4, False),
    ],
)
def test_dask_compute_graph(df, scheduler, num_workers, pairdensities):
    dreport = summarise(
        df,
        scheduler=scheduler,
        num_workers=num_workers,
        pairdensities=pairdensities,
    )._report
    fname = None
    if scheduler == "multiprocessing" and num_workers is None:
        fname = "{}/test_results/report_test_data_{}.json".format(
            dirname, "mp"
        )
    assert dreport["_lens_version"] == __version__
    if not pairdensities:
        assert dreport["pairdensity"] == {"_columns": [], "_run_time": 0.0}

    serialize_full_report(dreport, fname=fname)


def test_empty_df():
    empty_df = pd.DataFrame()
    with pytest.raises(EmptyDataFrameError):
        summarise(empty_df)


@pytest.fixture
def small_df():
    N = 100
    df = pd.DataFrame.from_dict(
        {"foo": np.random.randn(N), "bar": np.random.randint(10, size=N)}
    )
    return df


def test_string_num_cpus_env(small_df, monkeypatch):
    monkeypatch.setenv("NUM_CPUS", "not-an-int")
    ls = summarise(small_df)
    assert set(ls._report["_columns"]) == set(small_df.columns)


def test_int_num_cpus_env(small_df, monkeypatch):
    num_cpus_env = 2
    monkeypatch.setenv("NUM_CPUS", str(num_cpus_env))
    ls = summarise(small_df)
    assert set(ls._report["_columns"]) == set(small_df.columns)
