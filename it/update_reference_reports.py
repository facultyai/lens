from __future__ import print_function

import boto3
import os

S3 = boto3.client("s3")
BUCKET = "asi-lens-test-data"


def check_report_bom(name):
    """Check whether the report has Byte Order Marks

    Parameters
    ----------
    name : str
        Filename of the report.
    """
    with open(name, "r") as f:
        report_json = f.read()
    if report_json.count("\\ufeff"):
        print(
            "  WARNING: {} Byte-order-Marks found in report! This might"
            " provoke failures on codeship. Have you checked that the csv"
            " is encoded in UTF8 without BOM?".format(
                report_json.count("\\ufeff")
            )
        )


if __name__ == "__main__":
    from test_regression import datasets

    for dataset in datasets:
        report_name = dataset.replace(".csv", ".json")
        report_path = os.path.join("generated_reports", report_name)
        check_report_bom(report_path)
        S3.upload_file(
            report_path, Bucket=BUCKET, Key="output/{}".format(report_name)
        )
