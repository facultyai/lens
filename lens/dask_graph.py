"""Build a Dask graph of Summary computation"""

import itertools
import pandas as pd
from dask.delayed import delayed, Delayed
from . import metrics


def _nested_merge(first, second, path=None):
    """Merge two nested dictionaries into a single dictionary.

    Parameters
    ----------
    first : dict
        The first dictionary.
    second : dict
        The second dictionary.
    path : TODO
        TODO

    Returns
    -------
    dict
        The merged dictionary.
    """
    if path is None:
        path = []
    for key in second:
        if key in first:
            if isinstance(first[key], dict) and isinstance(second[key], dict):
                _nested_merge(first[key], second[key], path + [str(key)])
            elif first[key] == second[key]:
                pass  # Same leaf value.
            else:
                raise Exception(
                    "Conflict at {}".format(".".join(path + [str(key)]))
                )
        else:
            first[key] = second[key]
    return first


@delayed(pure=True)
def _join_dask_results(results):
    """Join a list of column-wise results into a single dictionary.

    The `_run_time` and `_columns` keys are appended to, whilst other
    keys are merged.

    Parameters
    ----------
    results : list
        List of Dask results dictionaries to join.
    """
    report = {"_run_time": 0.0, "_columns": []}

    for result in results:
        if isinstance(result, Delayed):
            result = result.compute()
        if result is not None:
            report["_run_time"] += result["_run_time"]
            report["_columns"] += result["_columns"]
            columns = result.keys()
            report = _nested_merge(
                report,
                {
                    column: result[column]
                    for column in columns
                    if column not in ["_columns", "_run_time"]
                },
            )

    report["_columns"] = sorted(list(set(report["_columns"])))

    return report


def create_dask_graph(df, pairdensities=True):
    """Create a Dask graph for executing the summary generation.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame for which to generate the summary.

    pairdensities : bool, optional
        Whether to compute the pairdensity estimation between all pairs of
        numerical columns. For most datasets, this is the most expensive
        computation. Default is True.

    Returns
    -------
    dict
        The generated data summary.
    """
    # Create a series for each column in the DataFrame.
    columns = df.columns
    df = delayed(df)
    cols = {k: delayed(df.get)(k) for k in columns}

    # Create the delayed reports using Dask.
    row_c = delayed(metrics.row_count)(df)

    cprops = {k: delayed(metrics.column_properties)(cols[k]) for k in columns}
    joined_cprops = _join_dask_results(list(cprops.values()))

    freqs = {
        k: delayed(metrics.frequencies)(cols[k], cprops[k]) for k in columns
    }
    joined_freqs = _join_dask_results(list(freqs.values()))

    csumms = {
        k: delayed(metrics.column_summary)(cols[k], cprops[k]) for k in columns
    }
    joined_csumms = _join_dask_results(list(csumms.values()))

    out = {k: delayed(metrics.outliers)(cols[k], csumms[k]) for k in columns}
    joined_outliers = _join_dask_results(list(out.values()))

    corr = delayed(metrics.correlation)(df, joined_cprops)

    pdens_results = []
    if pairdensities:
        for col1, col2 in itertools.combinations(columns, 2):
            pdens_df = delayed(pd.concat)([cols[col1], cols[col2]], axis=1)
            pdens_cp = {k: cprops[k] for k in [col1, col2]}
            pdens_cs = {k: csumms[k] for k in [col1, col2]}
            pdens_fr = {k: freqs[k] for k in [col1, col2]}
            pdens = delayed(metrics.pairdensity)(
                pdens_df, pdens_cp, pdens_cs, pdens_fr
            )
            pdens_results.append(pdens)

    joined_pairdensities = _join_dask_results(pdens_results)

    # Join the delayed reports per-metric into a dictionary.
    dask_dict = delayed(dict)(
        row_count=row_c,
        column_properties=joined_cprops,
        frequencies=joined_freqs,
        column_summary=joined_csumms,
        outliers=joined_outliers,
        correlation=corr,
        pairdensity=joined_pairdensities,
        _columns=list(columns),
    )

    return dask_dict
