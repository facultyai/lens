from __future__ import print_function

import json

import boto3
import lens
import numpy as np
import pandas as pd
import pytest
import os
import inspect

S3 = boto3.client("s3")
BUCKET = "asi-lens-test-data"

datasets = [
    "room_occupancy.csv",
    "artworks-5k.csv",
    "air-quality-london-time-of-day.csv",
    "momaExhibitions-5k.csv",
    "noheader.csv",
    "monthly-milk-production.csv",
    "customer-data.csv",
]

dirname = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
result_dir = os.path.join(dirname, "generated_reports")

if not os.path.exists(result_dir):
    os.mkdir(result_dir)


@pytest.mark.parametrize("input_", datasets)
def test_summary_regression(input_):
    # load the input into a pandas dataframe
    df = pd.read_csv("s3://{}/input/{}".format(BUCKET, input_))

    # run the lens summarise method
    summary = lens.summarise(df)

    # Save generated report
    summary.to_json(os.path.join(result_dir, input_.replace(".csv", ".json")))

    # load the expected output file into a summary object
    output = input_.replace(".csv", ".json")
    s3_summary = read_s3_file(BUCKET, "output/{}".format(output))[
        "Body"
    ].read()

    if isinstance(s3_summary, bytes):
        s3_summary = s3_summary.decode("utf-8")

    expected_summary = json.loads(s3_summary)

    # list of keys to ignore from the response because they are
    # probablistically generated
    exclude = [
        "_run_time",
        "tdigest",
        "density",
        "bw",
        "logtrans_IQR",
        "kde",
        "_lens_version",
    ]

    diffs = find_diff(
        json.loads(json.dumps(summary._report)), expected_summary, exclude
    )

    for diff in diffs:
        print(diff)

    if len(diffs):
        # Save expected report to check the differences manually if needed
        exp_name = os.path.join(
            result_dir, output.replace(".json", "-expected.json")
        )
        with open(exp_name, "w") as f:
            f.write(s3_summary)

    # compare the input and output summary objects
    assert len(diffs) == 0


def read_s3_file(bucket, key):
    return S3.get_object(Bucket=BUCKET, Key=key)


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
