import datetime as dt
import os
import inspect
import random
import string

from lens.dask_graph import create_dask_graph

import numpy as np
import pandas as pd
from scipy import stats
import pytest

np.random.seed(4712)

dirname = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)


@pytest.fixture(scope="module", params=[10, 60, 500, 2000])
def df(request):
    nrows = request.param
    n1 = np.random.randn(nrows) * 3 + 20.
    n2 = np.random.randn(nrows) * 5 + 30.
    poisson = np.random.poisson(10, nrows)
    items = [
        ("normal", n1 + n2),
        ("normal2", n1 - n2),
        ("uniform", np.random.random(nrows)),
        ("lognormal", stats.lognorm.rvs(5, scale=10, size=nrows)),
        ("poisson", poisson),
        ("categorical13", gen_poisson_distributed_categorical_data(13, nrows)),
        ("categorical5", gen_uniformly_distributed_categorical_data(5, nrows)),
        ("categorical2", np.random.randint(0, 2, nrows)),
        ("categoricalint", gen_categoricalint_with_no_twos(nrows)),
        ("ID", ["ID{}".format(x) for x in range(int(1e3), int(1e3 + nrows))]),
        ("datetimes", gen_datetime_strings(nrows)),
        ("dates", gen_date_strings(nrows)),
        ("times", gen_time_strings(nrows)),
        ("nulls", [np.nan] * nrows),
    ]

    df = pd.DataFrame.from_dict(dict(items))

    # sprinkle nrows/50 nulls
    ncols = len(df.columns)
    ii = np.random.randint(0, ncols, int((nrows * ncols) / 50))
    jj = np.random.randint(0, nrows, int((nrows * ncols) / 50))
    for i, j in zip(ii, jj):
        df.loc[j, list(df.columns)[i]] = None

    # No nulls in poissonint to avoid casting as floats
    df["poissonint"] = poisson
    # Add column that is strictly correlated with a float column
    df["normalcorr"] = n1 + n2
    # Add a column that has values where normal has nulls
    df["antinormal"] = np.where(df.normal.isnull(), n1 + n2, np.nan)

    df.to_csv(dirname + "/test_results/test_data.csv", index=False)

    return df


def gen_categoricalint_with_no_twos(nrows):
    values = np.random.randint(0, 6, nrows)
    values[values == 2] = 5
    return values


def gen_poisson_distributed_categorical_data(ncategories, size):
    categories = [
        str(i) + "".join(random.sample(string.ascii_letters, 4))
        for i in range(ncategories)
    ]
    random_samples = [
        np.random.poisson(ncategories / 2.0) for i in range(size)
    ]
    truncated_random_samples = [
        max(min(0, sample), ncategories - 1) for sample in random_samples
    ]
    sampled_categories = [
        categories[sample] for sample in truncated_random_samples
    ]
    return sampled_categories


def gen_uniformly_distributed_categorical_data(ncategories, size):
    categories = [
        str(i) + "".join(random.sample(string.ascii_letters, 4))
        for i in range(ncategories)
    ]
    random_samples = np.random.randint(0, len(categories), size=size)
    truncated_random_samples = [
        max(min(0, sample), ncategories - 1) for sample in random_samples
    ]
    sampled_categories = [
        categories[sample] for sample in truncated_random_samples
    ]
    return sampled_categories


def gen_date_strings(size):
    datetimes = gen_datetimes(size)
    date_strings = [datetime.date().isoformat() for datetime in datetimes]
    return date_strings


def gen_time_strings(size):
    datetimes = gen_datetimes(size)
    date_strings = [datetime.time().isoformat() for datetime in datetimes]
    return date_strings


def gen_datetime_strings(size):
    datetimes = gen_datetimes(size)
    datetime_strings = [datetime.isoformat() for datetime in datetimes]
    return datetime_strings


def gen_datetimes(size):
    timestamps = np.linspace(0, 86400 * 365 * 40, size)
    datetimes = [dt.datetime.fromtimestamp(ts) for ts in timestamps]
    return datetimes


@pytest.fixture(scope="module")
def report(df):
    # Get a dict report by not calling summarise
    report = create_dask_graph(df).compute(scheduler="multiprocessing")

    return report
