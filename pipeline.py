"""
pipeline.py -- ER Surge & Bed-Risk Advisor

Shared feature engineering + risk scoring logic. Written against the plain
pandas API only (groupby, transform, rolling, merge) so the exact same
function runs correctly whether `pandas` in this process is the real thing
or has been replaced by cudf.pandas's accelerator -- no code here should
ever need to change between the two.

This is intentionally the single source of truth: the Streamlit app,
the benchmark script, and the notebook all call run_feature_pipeline()
from here rather than keeping three copies in sync.
"""

import pandas as pd
import numpy as np

# Bed capacity per department. The real historical dataset has no bed-count
# column, so this is an explicit modeling assumption -- documented here
# rather than buried in a magic number. Tune these to make the demo's
# occupancy story land the way you want it to.
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

RISK_WEIGHTS = {
    "risk_wait": 0.25,
    "risk_load": 0.25,
    "risk_acuity": 0.20,
    "risk_complexity": 0.10,
    "risk_occupancy": 0.20,
}


def _normalize(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi - lo == 0:
        return s * 0
    return (s - lo) / (hi - lo)


def run_feature_pipeline(df: pd.DataFrame, bed_capacity: dict = None) -> pd.DataFrame:
    """
    Full feature engineering + risk scoring over the ENTIRE accumulated frame.

    This recomputes from scratch every time it's called -- no incremental
    update logic. That's a deliberate choice: it keeps the code simple, and
    it's what makes row count (and therefore GPU acceleration) meaningful.
    A live app calling this on 5 new rows would barely notice a GPU; calling
    it on a 300k-row accumulated operational history is where cuDF matters.

    Expects columns: patient_id, arrival_time, department, age,
    wait_time_min, admitted, case_mgmt_flag, is_critical, current_status,
    bed_id (bed_id/current_status/is_critical may be absent for pure
    historical data -- filled with safe defaults below).
    """
    bed_capacity = bed_capacity or DEPARTMENT_BED_CAPACITY
    d = df.copy()

    # Safe defaults for columns that only exist once live simulation starts,
    # so this function also works unmodified on the plain historical CSV.
    if "is_critical" not in d.columns:
        d["is_critical"] = 0
    if "current_status" not in d.columns:
        d["current_status"] = "Discharged"
    if "bed_id" not in d.columns:
        d["bed_id"] = None

    d = d.sort_values(["department", "arrival_time"]).reset_index(drop=True)

    d["hour"] = pd.to_datetime(d["arrival_time"]).dt.floor("h")

    # NOTE: rolling-per-group is written as groupby(...).rolling(...) rather
    # than groupby(...).transform(lambda s: s.rolling(...)) on purpose --
    # the transform+lambda form wraps a generic Python callable that
    # cudf.pandas is more likely to silently fall back to CPU for. The
    # direct .rolling() chain is the pattern cuDF actually vectorizes on
    # the GPU, which matters since the whole point here is to accelerate.

    # --- Wait-time pressure ---
    d["rolling_wait_avg_10"] = (
        d.groupby("department")["wait_time_min"]
         .rolling(window=10, min_periods=1).mean()
         .reset_index(level=0, drop=True)
    )

    # --- Relative arrival load (this hour vs. that department's own baseline) ---
    arrivals_per_hour = (
        d.groupby(["department", "hour"]).size().rename("arrivals_this_hour").reset_index()
    )
    d = d.merge(arrivals_per_hour, on=["department", "hour"], how="left")
    dept_baseline = (
        arrivals_per_hour.groupby("department")["arrivals_this_hour"]
        .mean().rename("dept_baseline_arrivals").reset_index()
    )
    d = d.merge(dept_baseline, on="department", how="left")
    d["relative_load"] = d["arrivals_this_hour"] / d["dept_baseline_arrivals"]

    # --- Acuity: admission rate blended with explicit critical flag ---
    d["rolling_admit_rate_10"] = (
        d.groupby("department")["admitted"]
         .rolling(window=10, min_periods=1).mean()
         .reset_index(level=0, drop=True)
    )
    d["rolling_critical_share_10"] = (
        d.groupby("department")["is_critical"]
         .rolling(window=10, min_periods=1).mean()
         .reset_index(level=0, drop=True)
    )
    d["acuity_signal"] = 0.5 * d["rolling_admit_rate_10"] + 0.5 * d["rolling_critical_share_10"]

    # --- Complexity proxy ---
    d["rolling_cm_rate_10"] = (
        d.groupby("department")["case_mgmt_flag"]
         .rolling(window=10, min_periods=1).mean()
         .reset_index(level=0, drop=True)
    )

    # --- Bed occupancy (only meaningful once bed_id / current_status are populated) ---
    active_mask = d["current_status"].isin(["Waiting", "Under Treatment", "Admitted"])
    occ_by_dept = (
        d[active_mask].groupby("department")["bed_id"].nunique().rename("beds_occupied")
    )
    cap_series = pd.Series(bed_capacity, name="beds_total")
    occ_table = pd.concat([occ_by_dept, cap_series], axis=1).fillna(0)
    occ_table["occupancy_ratio"] = (occ_table["beds_occupied"] / occ_table["beds_total"]).clip(upper=1.5)
    d = d.merge(
        occ_table["occupancy_ratio"].rename("dept_occupancy_ratio"),
        left_on="department", right_index=True, how="left"
    )
    d["dept_occupancy_ratio"] = d["dept_occupancy_ratio"].fillna(0)

    # --- Normalize + combine ---
    d["risk_wait"] = _normalize(d["rolling_wait_avg_10"])
    d["risk_load"] = _normalize(d["relative_load"])
    d["risk_acuity"] = _normalize(d["acuity_signal"])
    d["risk_complexity"] = _normalize(d["rolling_cm_rate_10"])
    d["risk_occupancy"] = _normalize(d["dept_occupancy_ratio"])

    d["bed_risk_score"] = sum(d[col] * w for col, w in RISK_WEIGHTS.items())

    high_cut = d["bed_risk_score"].quantile(0.90)
    med_cut = d["bed_risk_score"].quantile(0.65)
    d["risk_tier"] = pd.cut(
        d["bed_risk_score"], bins=[-0.01, med_cut, high_cut, 2.0],
        labels=["Low", "Medium", "High"]
    )

    return d


def department_summary(scored_df: pd.DataFrame, bed_capacity: dict = None) -> pd.DataFrame:
    """
    Collapse the scored, row-level dataframe into one row per department --
    this is what the dashboard actually renders. Uses only the most recent
    arrival's score per department as "current state" (the score already
    reflects the rolling/accumulated history via run_feature_pipeline).
    """
    bed_capacity = bed_capacity or DEPARTMENT_BED_CAPACITY
    latest = (
        scored_df.sort_values("arrival_time")
                 .groupby("department")
                 .tail(1)
                 .set_index("department")
    )
    summary = pd.DataFrame({
        "occupancy_pct": (latest["dept_occupancy_ratio"] * 100).round(1),
        "avg_wait_min": latest["rolling_wait_avg_10"].round(1),
        "bed_risk_score": latest["bed_risk_score"].round(3),
        "risk_tier": latest["risk_tier"],
    })
    for dept in bed_capacity:
        if dept not in summary.index:
            summary.loc[dept] = [0.0, 0.0, 0.0, "Low"]
    return summary.sort_values("bed_risk_score", ascending=False)


def recommend_action(dept: str, row: pd.Series) -> list:
    """
    Simple, explainable rule-based recommendations -- deliberately not a
    black box. A charge nurse needs to be able to see exactly why a
    recommendation fired.
    """
    actions = []
    if row["risk_tier"] == "High":
        if row["occupancy_pct"] >= 90:
            actions.append(f"Open overflow beds -- {dept} at {row['occupancy_pct']:.0f}% capacity")
        if row["avg_wait_min"] >= 45:
            actions.append(f"Request additional staff -- avg wait {row['avg_wait_min']:.0f} min in {dept}")
        actions.append(f"Prioritize triage review for incoming {dept} patients")
    elif row["risk_tier"] == "Medium":
        actions.append(f"Monitor {dept} closely -- trending toward elevated risk")
    else:
        actions.append("No action required")
    return actions