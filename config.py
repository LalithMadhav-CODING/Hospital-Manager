"""
config.py -- single place for paths and the CPU/GPU switch.

Nothing else in the project should hardcode a path or the USE_CUDF flag.
When you're ready to actually run the GPU path (Colab, real hardware),
this is the only line that changes.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "Hospital_ER_Data.csv"
BENCHMARK_RESULTS_PATH = DATA_DIR / "benchmark_results.csv"

# Flip to True only in an environment with an NVIDIA GPU + cudf-cu12 installed
# (e.g. Colab). Free-tier deployment targets (Streamlit Community Cloud,
# Cloud Run free tier) have no GPU -- keep this False there and say so
# in the deck/README rather than hiding it.
USE_CUDF = False

# Bed capacity per department -- documented modeling assumption, since the
# real historical dataset has no bed-count column. Lives here (not buried
# in pipeline.py) so it's the one place to tune for the demo.
DEPARTMENT_BED_CAPACITY = {
    "General ER": 40,
    "General Practice": 15,
    "Orthopedics": 12,
    "Physiotherapy": 6,
    "Cardiology": 10,
    "Neurology": 8,
    "Gastroenterology": 8,
    "Renal": 6,
}

# Event-driven recalculation thresholds
RECALC_EVERY_N_ARRIVALS = 5
RECALC_EVERY_N_MINUTES = 5
RECALC_OCCUPANCY_THRESHOLD = 0.90

# Baseline "operational history" size seeded at app startup
DEFAULT_BASELINE_ROWS = 50_000