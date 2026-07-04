"""
simulate.py -- ER Surge & Bed-Risk Advisor

Generates:
  1. A large synthetic "operational history" baseline (20k-500k+ rows) by
     bootstrap-sampling the real Hospital_ER_Data.csv distributions with
     jitter -- NOT row duplication. This is the accumulated history the
     live app's risk engine reasons against, and the row count that makes
     GPU acceleration meaningful.
  2. Live patient arrival events for the Streamlit simulation mode, with
     realistic burst patterns and forced critical-patient injection for
     demo control.

Keep this separate from pipeline.py: this file only ever produces raw,
transactional-looking rows (never a stored risk score or recommendation --
those are always computed live from pipeline.py).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools

DEPARTMENTS = [
    "General ER", "General Practice", "Orthopedics", "Physiotherapy",
    "Cardiology", "Neurology", "Gastroenterology", "Renal",
]

ARRIVAL_MODES = ["Walk-in", "Ambulance", "Referral"]
ARRIVAL_MODE_WEIGHTS = [0.65, 0.20, 0.15]

# Ambulance arrivals are more likely to be critical -- a deliberate,
# documented modeling choice, not an arbitrary number.
CRITICAL_PROB_BY_MODE = {"Walk-in": 0.03, "Ambulance": 0.35, "Referral": 0.08}

_id_counter = itertools.count(1)


def _reset_id_counter(start=1):
    global _id_counter
    _id_counter = itertools.count(start)


def load_historical(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    df = pd.DataFrame()
    df["arrival_time"] = pd.to_datetime(raw["Patient Admission Date"], format="%d-%m-%Y %H:%M")
    df["department"] = raw["Department Referral"].fillna("General ER")
    df["age"] = raw["Patient Age"]
    df["wait_time_min"] = raw["Patient Waittime"].astype(float)
    df["admitted"] = raw["Patient Admission Flag"].astype(int)
    df["case_mgmt_flag"] = raw["Patients CM"].astype(int)
    return df


def _empirical_sample(historical: pd.DataFrame, department: str, column: str, rng, jitter=0.0):
    """Bootstrap a single value from the real distribution for a given
    department + column, with optional small multiplicative jitter so
    synthetic rows aren't exact duplicates of real ones."""
    pool = historical.loc[historical["department"] == department, column]
    if pool.empty:
        pool = historical[column]
    val = rng.choice(pool.values)
    if jitter:
        val = val * (1 + rng.uniform(-jitter, jitter))
    return val


def generate_baseline(
    historical: pd.DataFrame,
    n_rows: int = 300_000,
    start: datetime = None,
    span_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a large synthetic operational-history dataframe by bootstrap-
    sampling the real dataset's per-department distributions and spreading
    arrivals across a longer synthetic time window. Preserves department
    mix, age range, wait-time shape, admission rate, and case-mgmt rate
    from the real data -- just at the scale the GPU benchmark needs.
    """
    rng = np.random.default_rng(seed)
    start = start or (datetime.now() - timedelta(days=span_days))

    dept_counts = historical["department"].value_counts(normalize=True)
    depts = rng.choice(dept_counts.index, size=n_rows, p=dept_counts.values)

    minute_offsets = rng.integers(0, span_days * 24 * 60, size=n_rows)
    arrival_times = [start + timedelta(minutes=int(m)) for m in minute_offsets]

    ages = np.empty(n_rows)
    waits = np.empty(n_rows)
    admits = np.empty(n_rows, dtype=int)
    cms = np.empty(n_rows, dtype=int)

    for dept in dept_counts.index:
        mask = depts == dept
        n = mask.sum()
        if n == 0:
            continue
        ages[mask] = rng.choice(historical.loc[historical.department == dept, "age"].values, size=n)
        waits[mask] = rng.choice(historical.loc[historical.department == dept, "wait_time_min"].values, size=n) \
            * (1 + rng.uniform(-0.1, 0.1, size=n))
        admits[mask] = rng.choice(historical.loc[historical.department == dept, "admitted"].values, size=n)
        cms[mask] = rng.choice(historical.loc[historical.department == dept, "case_mgmt_flag"].values, size=n)

    arrival_modes = rng.choice(ARRIVAL_MODES, size=n_rows, p=ARRIVAL_MODE_WEIGHTS)
    critical = np.array([rng.random() < CRITICAL_PROB_BY_MODE[m] for m in arrival_modes]).astype(int)

    global _id_counter
    start_id = next(_id_counter)
    ids = [f"P{i:07d}" for i in range(start_id, start_id + n_rows)]
    _id_counter = itertools.count(start_id + n_rows)

    df = pd.DataFrame({
        "patient_id": ids,
        "arrival_time": arrival_times,
        "department": depts,
        "age": ages.round().astype(int),
        "arrival_mode": arrival_modes,
        "is_critical": critical,
        "wait_time_min": waits.round(1),
        "admitted": admits,
        "case_mgmt_flag": cms,
        "current_status": "Discharged",  # baseline history is all resolved
        "bed_id": None,
    })
    return df.sort_values("arrival_time").reset_index(drop=True)


def generate_arrival_batch(
    historical: pd.DataFrame,
    n: int = 1,
    now: datetime = None,
    force_critical: bool = False,
    force_department: str = None,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate n new live patient arrivals "right now", for the Streamlit
    simulation loop. Each call produces fresh rows to append to the
    accumulated live dataframe -- never mutates historical/baseline data.
    """
    rng = np.random.default_rng(seed)
    now = now or datetime.now()
    dept_counts = historical["department"].value_counts(normalize=True)

    rows = []
    for _ in range(n):
        dept = force_department or rng.choice(dept_counts.index, p=dept_counts.values)
        mode = rng.choice(ARRIVAL_MODES, p=ARRIVAL_MODE_WEIGHTS)
        is_critical = 1 if force_critical else int(rng.random() < CRITICAL_PROB_BY_MODE[mode])
        age = int(_empirical_sample(historical, dept, "age", rng))
        wait = round(float(_empirical_sample(historical, dept, "wait_time_min", rng, jitter=0.15)), 1)
        # critical patients are fast-tracked -- shorter wait, near-certain admission
        if is_critical:
            wait = round(wait * rng.uniform(0.2, 0.5), 1)
            admitted = 1
        else:
            admitted = int(_empirical_sample(historical, dept, "admitted", rng))
        cm_flag = int(_empirical_sample(historical, dept, "case_mgmt_flag", rng))

        rows.append({
            "patient_id": f"P{next(_id_counter):07d}",
            "arrival_time": now,
            "department": dept,
            "age": age,
            "arrival_mode": mode,
            "is_critical": is_critical,
            "wait_time_min": wait,
            "admitted": admitted,
            "case_mgmt_flag": cm_flag,
            "current_status": "Waiting",
            "bed_id": None,  # assigned by assign_beds()
        })
    return pd.DataFrame(rows)


def assign_beds(live_df: pd.DataFrame, bed_capacity: dict) -> pd.DataFrame:
    """
    Assign bed IDs to active (Waiting/Under Treatment/Admitted) patients up
    to each department's capacity. Patients beyond capacity are left
    unassigned (bed_id stays None) -- boarding with no bed is itself a
    strong, legible risk signal for the dashboard, not an error state.
    """
    d = live_df.copy()
    active = d["current_status"].isin(["Waiting", "Under Treatment", "Admitted"])
    for dept, capacity in bed_capacity.items():
        dept_mask = active & (d["department"] == dept)
        already_assigned = d.loc[dept_mask & d["bed_id"].notna(), "bed_id"]
        used_numbers = set()
        for b in already_assigned:
            try:
                used_numbers.add(int(str(b).split("-")[-1]))
            except ValueError:
                pass
        free_numbers = [i for i in range(1, capacity + 1) if i not in used_numbers]
        needs_bed_idx = d.loc[dept_mask & d["bed_id"].isna()].index
        for idx, num in zip(needs_bed_idx, free_numbers):
            prefix = "".join(w[0] for w in dept.split())[:3].upper()
            d.at[idx, "bed_id"] = f"{prefix}-{num}"
    return d


def should_recalculate(
    live_df: pd.DataFrame,
    last_recalc_count: int,
    last_recalc_time: datetime,
    bed_capacity: dict,
    every_n_arrivals: int = 5,
    every_n_minutes: int = 15,
    occupancy_threshold: float = 0.90,
) -> tuple[bool, str]:
    """
    Event-driven recalculation trigger, per the design principle: don't
    recompute on a fixed schedule, recompute when something operationally
    meaningful happened. Returns (should_run, reason).
    """
    n_new = len(live_df) - last_recalc_count
    if n_new <= 0:
        return False, "no new arrivals"

    recent = live_df.tail(n_new)
    if recent["is_critical"].sum() > 0:
        return True, "critical patient arrived"

    active = live_df[live_df["current_status"].isin(["Waiting", "Under Treatment", "Admitted"])]
    for dept, capacity in bed_capacity.items():
        occ = (active["department"] == dept).sum() / capacity
        if occ >= occupancy_threshold:
            return True, f"{dept} occupancy crossed {occupancy_threshold:.0%}"

    if n_new >= every_n_arrivals:
        return True, f"{every_n_arrivals}+ new arrivals"

    if last_recalc_time and (datetime.now() - last_recalc_time) >= timedelta(minutes=every_n_minutes):
        return True, f"{every_n_minutes} minutes elapsed"

    return False, "no trigger condition met"