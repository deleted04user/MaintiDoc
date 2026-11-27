"""Log aggregate retrieval measurements locally or to MLFLOW_TRACKING_URI."""
import argparse
import json
import os
from pathlib import Path

import mlflow


def record(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    uri = os.getenv("MLFLOW_TRACKING_URI", (Path(".maintidoc/mlruns").resolve()).as_uri())
    if uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("MaintiDoc")
    with mlflow.start_run(run_name=path.stem) as run:
        mlflow.log_param("encoder", report["encoder"])
        mlflow.log_metrics(report["mean"])
        print(run.info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    record(parser.parse_args().report)
