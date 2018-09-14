import os
import inspect

import pytest
import numpy as np
import numpy.testing
import scipy.stats
import pandas as pd
import json

from lens import Summary, summarise

from fixtures import df, report  # noqa

dirname = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)


@pytest.fixture(scope="function")
def ls(report):
    return Summary(report)


# VZ: I have not managed to get the below test not to mutate the report
# fixture, so subsequent tests fail if this is run. Disabling for now.
# def test_report_validation(report):
#     # load it into the class
#     Summary(report)
#     r = report.copy()
#
#     # Test that it fails on missing data
#     for metric in ['frequencies', 'column_summary', 'outliers']:
#         r[metric].pop(r[metric]['_columns'][0])
#         with pytest.raises(LensSummaryError):
#             Summary(r)


def test_columns_method(report, ls):
    assert set(ls.columns) == set(report["_columns"])


def test_row_count_method(report, ls):
    assert report["row_count"]["total"] == ls.rows
    assert report["row_count"]["unique"] == ls.rows_unique


def test_summary_method(report, ls):
    for col in ls.columns:
        summary = ls.summary(col)
        assert summary["name"] == col
        for k in ["nulls", "notnulls", "unique", "dtype"]:
            assert summary[k] == report["column_properties"][col][k]

        assert (summary["desc"] == "categorical") == report[
            "column_properties"
        ][col]["is_categorical"]

        assert (summary["desc"] == "numeric") == (
            report["column_properties"][col]["numeric"]
            and not report["column_properties"][col]["is_categorical"]
        )


def test_numeric_details(report, ls):
    num_cols = [
        col
        for col in ls.columns
        if report["column_properties"][col]["numeric"]
    ]

    metrics = ["min", "max", "mean", "median", "std", "sum"]

    for col in num_cols:
        details = ls.details(col)
        for m in metrics:
            if not np.isnan(details[m]):
                assert details[m] == report["column_summary"][col][m]


def test_categorical_details(report, ls):
    cat_cols = [
        col
        for col in ls.columns
        if report["column_properties"][col]["is_categorical"]
    ]

    for col in cat_cols:
        details = ls.details(col)
        for category in report["frequencies"][col].keys():
            assert (
                details["frequencies"][category]
                == report["frequencies"][col][category]
            )


def test_histogram(report, ls):
    num_cols = [
        col
        for col in ls.columns
        if report["column_properties"][col]["numeric"]
    ]

    for col in num_cols:
        histogram = ls.histogram(col)
        for key, actual in zip(["counts", "bin_edges"], histogram):
            assert np.allclose(
                report["column_summary"][col]["histogram"][key], actual
            )


def test_kde(report, ls):
    num_cols = [
        col
        for col in ls.columns
        if report["column_properties"][col]["numeric"]
    ]

    for col in num_cols:
        kde = ls.kde(col)
        for key, actual in zip(["x", "y"], kde):
            assert np.allclose(
                report["column_summary"][col]["kde"][key], actual
            )


@pytest.mark.parametrize(
    "col1, col2",
    [
        ("normal", "poisson"),
        ("normal", "lognormal"),
        ("normal", "categorical5"),
        ("categorical5", "categorical13"),
    ],
)
def test_pair_details_pairdensity(report, ls, col1, col2):
    details = ls.pair_details(col1, col2)

    for col, key in zip([col1, col2], ["x", "y"]):
        if col in report["column_summary"].keys():
            # Test that logtrans matches scale.
            assert report["column_summary"][col]["logtrans"] == (
                details["pairdensity"][key + "_scale"] == "log"
            )
            # Test that min/max match range of coordinates.
            assert np.allclose(
                report["column_summary"][col]["min"],
                np.min(details["pairdensity"][key]),
            )
            assert (
                np.max(details["pairdensity"][key])
                <= report["column_summary"][col]["max"]
            )

    details_transposed = ls.pair_details(col2, col1)
    assert np.allclose(
        details["pairdensity"]["density"],
        details_transposed["pairdensity"]["density"].T,
    )


@pytest.mark.parametrize(
    "col1, col2", [("normal", "poisson"), ("normal", "lognormal")]
)
def test_pair_details_correlation(report, ls, col1, col2):
    details = ls.pair_details(col1, col2)
    details_transposed = ls.pair_details(col1, col2)
    idx = [
        report["correlation"]["_columns"].index(col) for col in [col1, col2]
    ]

    for coeff in ["spearman", "pearson"]:
        assert np.allclose(
            report["correlation"][coeff][idx[0]][idx[1]],
            details["correlation"][coeff],
        )
        assert np.allclose(
            details["correlation"][coeff],
            details_transposed["correlation"][coeff],
        )


def test_pair_details_empty(ls):
    # Test that non-numeric pairs return an empty dict without raising
    # exceptions.
    details = ls.pair_details("normal", "datetimes")
    assert len(details.keys()) == 0


def test_pair_details_same_column(ls):
    with pytest.raises(ValueError):
        ls.pair_details("normal", "normal")


@pytest.mark.parametrize(
    "col1, col2", [("normal", "lognormal"), ("normal", "normal")]
)
def test_correlation_matrix(report, ls, col1, col2):
    columns, correlation_matrix = ls.correlation_matrix()
    index_column1 = columns.index(col1)
    index_column2 = columns.index(col2)
    correlation_value = (
        1
        if col1 == col2
        else (ls.pair_details(col1, col2)["correlation"]["spearman"])
    )
    assert (
        correlation_matrix[index_column1, index_column2]
        == correlation_matrix[index_column2, index_column1]
        == correlation_value
    )


def test_correlation_matrix_one_column():
    column_values = np.random.ranf(size=200)
    df = pd.DataFrame.from_dict({"a": column_values})
    summary = summarise(df)
    columns, correlation_matrix = summary.correlation_matrix()
    assert columns == ["a"]
    assert correlation_matrix.shape == (1, 1)
    numpy.testing.assert_approx_equal(correlation_matrix[0, 0], 1.0)


def test_correlation_matrix_two_columns():
    column1_values = np.random.ranf(size=200)
    column2_values = np.random.ranf(size=200)
    df = pd.DataFrame.from_dict({"a": column1_values, "b": column2_values})
    summary = summarise(df)
    columns, correlation_matrix = summary.correlation_matrix()
    assert sorted(columns) == ["a", "b"]
    numpy.testing.assert_approx_equal(correlation_matrix[0, 0], 1.0)
    numpy.testing.assert_approx_equal(correlation_matrix[1, 1], 1.0)
    off_diagonal_term = scipy.stats.spearmanr(df["a"], df["b"]).correlation
    numpy.testing.assert_approx_equal(
        correlation_matrix[1, 0], off_diagonal_term
    )
    numpy.testing.assert_approx_equal(
        correlation_matrix[0, 1], off_diagonal_term
    )


def test_correlation_matrix_three_columns():
    column_values = [np.random.ranf(size=200) for i in range(3)]
    column_headers = ["a", "b", "c"]
    df = pd.DataFrame.from_dict(dict(zip(column_headers, column_values)))
    summary = summarise(df)
    columns, correlation_matrix = summary.correlation_matrix()
    assert sorted(columns) == column_headers

    for i, first_column in enumerate(columns):
        for j, second_column in enumerate(columns):
            expected = scipy.stats.spearmanr(
                df[first_column], df[second_column]
            ).correlation
            actual = correlation_matrix[i, j]
            numpy.testing.assert_approx_equal(expected, actual)


def test_json_roundtrip(ls):
    # Run reference report through JSON roundtrip for comparison
    original_report = json.loads(json.dumps(ls._report))
    string_report = json.loads(ls.to_json())

    filename = "test-report.json"

    # Test filename roundtrip
    ls.to_json(filename)
    file_report = Summary.from_json(filename)._report

    # Test buffer roundtrip
    with open(filename, "w") as f:
        ls.to_json(f)

    with open(filename, "r") as f:
        buffer_report = Summary.from_json(f)._report

    os.remove(filename)

    for json_report in [string_report, file_report, buffer_report]:
        diffs = find_diff(original_report, json_report)
        for diff in diffs:
            print(diff)

        assert len(diffs) == 0


def find_diff(d1, d2, exclude=[], path="", update_path=True):
    diffs = []
    for k in d1.keys():
        if k in exclude:
            continue

        if k not in d2:
            msg = "{} :\n {} as key not in d2".format(path, k)
            diffs.append(msg)
        else:
            new_path = path
            if update_path:
                if new_path == "":
                    new_path = k
                else:
                    new_path = new_path + "->" + k

            if isinstance(d1[k], dict):
                diffs = diffs + find_diff(d1[k], d2[k], exclude, new_path)
            elif isinstance(d1[k], list):
                # convert the list to a dict using the index as the key.
                diffs = diffs + find_diff(
                    list_to_dict(d1[k]),
                    list_to_dict(d2[k]),
                    exclude,
                    new_path,
                    False,
                )
            else:
                a = d1[k]
                b = d2[k]
                if not isinstance(a, float) or not (
                    np.isnan(a) and np.isnan(b)
                ):
                    if isinstance(a, float):
                        if not np.allclose(a, b):
                            msg = "{} :\n - {} : {}\n + {} : {}".format(
                                path, k, a, k, b
                            )
                            diffs.append(msg)
                    elif a != b:
                        msg = "{} :\n - {} : {}\n + {} : {}".format(
                            path, k, a, k, b
                        )
                        diffs.append(msg)

    return diffs


def list_to_dict(list_):
    dict_ = {}
    for index, item in enumerate(list_):
        dict_[index] = item

    return dict_


# Tolerances for N=10k, taken from the TDigest test suite
tdigest_tol = {50: 0.02, 25: 0.015, 10: 0.01, 1: 0.005, 0.1: 0.001}

for k in list(tdigest_tol.keys()):
    tdigest_tol[100 - k] = tdigest_tol[k]


@pytest.mark.parametrize("column", ["normal", "lognormal", "poisson"])
def test_summary_cdf(ls, column):
    cdf = ls.cdf(column)

    # Set tolerance based on number of rows
    for p in ls._report["column_summary"][column]["percentiles"]:
        tol = tdigest_tol[p] * np.sqrt(10000 / ls.rows)
        x = ls._report["column_summary"][column]["percentiles"][p]
        assert np.allclose(p / 100., cdf(x), atol=tol, rtol=1)
