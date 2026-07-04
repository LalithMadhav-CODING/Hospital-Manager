"""
run_pipeline_backend.py

Standalone entry point that runs the ER pipeline under a chosen backend and
reports timing as JSON on stdout. Exists as a separate process specifically
because cudf.pandas patches the `pandas` module at import time -- that
can't be toggled inside an already-running app, so each backend gets a
fresh process instead.

Usage:
    python run_pipeline_backend.py --backend pandas --input baseline.parquet
    python run_pipeline_backend.py --backend cudf    --input baseline.parquet

The Streamlit app calls this via subprocess for the live CPU-vs-GPU demo
comparison. Requires a CUDA GPU + `pip install cudf-cu12` for --backend cudf
(e.g. in Colab) -- it will raise a clear error if cuDF isn't available
rather than silently falling back to pandas.
"""

import argparse
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pandas", "cudf"], required=True)
    parser.add_argument("--input", required=True, help="Path to a parquet/csv file with the accumulated live dataframe")
    args = parser.parse_args()

    if args.backend == "cudf":
        try:
            import cudf.pandas
            cudf.pandas.install()
        except ImportError:
            print(json.dumps({
                "backend": "cudf",
                "error": "cudf.pandas is not installed in this environment. "
                         "Run `pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12` "
                         "on a machine with an NVIDIA GPU (e.g. Colab with a T4/A100 runtime)."
            }))
            sys.exit(1)

    import pandas as pd
    from pipeline import run_feature_pipeline

    load_start = time.perf_counter()
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input, parse_dates=["arrival_time"])
    load_end = time.perf_counter()

    compute_start = time.perf_counter()
    scored = run_feature_pipeline(df)
    compute_end = time.perf_counter()

    print(json.dumps({
        "backend": args.backend,
        "n_rows": len(df),
        "load_seconds": round(load_end - load_start, 4),
        "compute_seconds": round(compute_end - compute_start, 4),
        "total_seconds": round(compute_end - load_start, 4),
        "risk_tier_counts": scored["risk_tier"].value_counts().to_dict(),
    }))


if __name__ == "__main__":
    main()