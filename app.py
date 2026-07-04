"""
app.py -- ER Surge & Bed-Risk Advisor (Streamlit)

Single-file app: Dashboard / Simulation / Benchmark as tabs, not separate
pages -- avoids page-navigation lag during a recorded demo and keeps all
state in one st.session_state block.

Design notes that matter for judges reading this file:
- Recalculation is event-driven (simulate.should_recalculate), not on a
  fixed timer -- the whole point of the "acceleration enables a decision"
  story.
- The pipeline always recomputes over the FULL accumulated dataframe
  (baseline + live arrivals), never an incremental delta -- that's what
  makes row count, and therefore GPU acceleration, actually meaningful.
- This deployed app runs the pandas path only. The cuDF path requires a
  real NVIDIA GPU (e.g. Colab), which free-tier hosting doesn't provide --
  the Benchmark tab is explicit about this rather than hiding it.
"""

import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st

import config
from pipeline import run_feature_pipeline, department_summary, recommend_action
from simulate import (
    load_historical, generate_baseline, generate_arrival_batch,
    assign_beds, should_recalculate, _reset_id_counter,
)

st.set_page_config(page_title="ER Surge & Bed-Risk Advisor", layout="wide")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

@st.cache_data
def _load_historical_cached():
    return load_historical(str(config.CSV_PATH))


def _log_event(message: str):
    st.session_state.event_log.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "message": message,
    })
    st.session_state.event_log = st.session_state.event_log[:30]


def _init_state():
    if "initialized" in st.session_state:
        return
    _reset_id_counter(1)
    hist = _load_historical_cached()
    baseline = generate_baseline(hist, n_rows=config.DEFAULT_BASELINE_ROWS, span_days=180, seed=1)
    st.session_state.historical = hist
    st.session_state.live_df = baseline
    st.session_state.last_recalc_count = 0
    st.session_state.last_recalc_time = None
    st.session_state.scored_df = None
    st.session_state.dashboard = None
    st.session_state.event_log = []
    st.session_state.last_recalc_reason = "initial load"
    st.session_state.initialized = True
    _log_event(f"Baseline seeded: {len(baseline):,} historical records")
    _recalculate(force=True, reason="initial load")


def _recalculate(force: bool = False, reason: str = None):
    live_df = st.session_state.live_df
    should, auto_reason = should_recalculate(
        live_df,
        st.session_state.last_recalc_count,
        st.session_state.last_recalc_time,
        config.DEPARTMENT_BED_CAPACITY,
        every_n_arrivals=config.RECALC_EVERY_N_ARRIVALS,
        every_n_minutes=config.RECALC_EVERY_N_MINUTES,
        occupancy_threshold=config.RECALC_OCCUPANCY_THRESHOLD,
    )
    if not (force or should):
        st.session_state.last_recalc_reason = f"skipped -- {auto_reason}"
        return False

    t0 = time.perf_counter()
    scored = run_feature_pipeline(live_df, config.DEPARTMENT_BED_CAPACITY)
    t1 = time.perf_counter()

    st.session_state.scored_df = scored
    st.session_state.dashboard = department_summary(scored, config.DEPARTMENT_BED_CAPACITY)
    st.session_state.last_recalc_count = len(live_df)
    st.session_state.last_recalc_time = datetime.now()
    st.session_state.last_recalc_seconds = t1 - t0
    st.session_state.last_recalc_reason = reason or auto_reason
    _log_event(f"Recalculated over {len(live_df):,} rows in {t1 - t0:.2f}s -- trigger: {st.session_state.last_recalc_reason}")
    return True


_init_state()


# ---------------------------------------------------------------------------
# Header / overall status banner
# ---------------------------------------------------------------------------

st.title("🏥 ER Surge & Bed-Risk Advisor")
st.caption("Decision support for the ED charge nurse: divert patients, call in staff, or open overflow beds -- based on live operational risk, not a static report.")

dash = st.session_state.dashboard
active = st.session_state.live_df
active_patients = active[active["current_status"].isin(["Waiting", "Under Treatment", "Admitted"])]

if dash is not None and len(dash):
    worst = dash.iloc[0]
    tier_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(worst["risk_tier"], "🟢")
    overall_label = {"High": "HIGH RISK", "Medium": "MODERATE", "Low": "NORMAL"}.get(worst["risk_tier"], "NORMAL")
else:
    tier_icon, overall_label, worst = "🟢", "NORMAL", None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Risk", f"{tier_icon} {overall_label}")
c2.metric("Avg Wait (worst dept)", f"{worst['avg_wait_min']:.0f} min" if worst is not None else "--")
c3.metric("Occupancy (worst dept)", f"{worst['occupancy_pct']:.0f}%" if worst is not None else "--")
c4.metric("Patients Active", f"{len(active_patients):,}")
c5.metric("Critical Active", int(active_patients["is_critical"].sum()) if len(active_patients) else 0)

if worst is not None:
    st.subheader("Recommendation" + (f" -- {worst.name}" if worst is not None else ""))
    for action in recommend_action(worst.name, worst):
        st.markdown(f"- {action}")

st.divider()

tab_dashboard, tab_simulation, tab_benchmark = st.tabs(["📊 Dashboard", "▶️ Simulation", "⚡ Benchmark"])


# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------

with tab_dashboard:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Department status")
        if dash is not None:
            display = dash.copy()
            display["risk_tier"] = display["risk_tier"].map(
                {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}
            )
            st.dataframe(
                display.rename(columns={
                    "occupancy_pct": "Occupancy %", "avg_wait_min": "Avg Wait (min)",
                    "bed_risk_score": "Risk Score", "risk_tier": "Tier",
                }),
                width='stretch',
            )
        st.caption(
            f"Computed over {st.session_state.last_recalc_count:,} accumulated records "
            f"in {st.session_state.get('last_recalc_seconds', 0):.2f}s (pandas) -- "
            f"last trigger: {st.session_state.last_recalc_reason}"
        )
        if st.button("🔄 Force recalculate now"):
            _recalculate(force=True, reason="manual refresh")
            st.rerun()

    with right:
        st.subheader("Event log")
        for e in st.session_state.event_log:
            st.text(f"[{e['time']}] {e['message']}")


# ---------------------------------------------------------------------------
# Simulation tab
# ---------------------------------------------------------------------------

with tab_simulation:
    st.subheader("Simulate live ED arrivals")
    st.caption(
        "New arrivals append to the same accumulated dataframe the dashboard reads from. "
        "Recalculation is event-driven -- watch the event log to see when it actually fires "
        "and why."
    )

    hist = st.session_state.historical
    dept_options = ["(random)"] + list(config.DEPARTMENT_BED_CAPACITY.keys())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Normal arrivals**")
        n_normal = st.number_input("Count", min_value=1, max_value=20, value=3, key="n_normal")
        if st.button("Add patients"):
            batch = generate_arrival_batch(hist, n=n_normal)
            st.session_state.live_df = assign_beds(
                pd.concat([st.session_state.live_df, batch], ignore_index=True),
                config.DEPARTMENT_BED_CAPACITY,
            )
            _log_event(f"{n_normal} new patient(s) arrived")
            _recalculate()
            st.rerun()

    with col2:
        st.markdown("**Critical patient**")
        crit_dept = st.selectbox("Department", dept_options, key="crit_dept")
        if st.button("🚨 Add critical patient"):
            dept = None if crit_dept == "(random)" else crit_dept
            batch = generate_arrival_batch(hist, n=1, force_critical=True, force_department=dept)
            st.session_state.live_df = assign_beds(
                pd.concat([st.session_state.live_df, batch], ignore_index=True),
                config.DEPARTMENT_BED_CAPACITY,
            )
            _log_event(f"🚨 Critical patient arrived -- {batch.iloc[0]['department']}")
            _recalculate(force=True, reason="critical patient arrived")
            st.rerun()

    with col3:
        st.markdown("**Ambulance surge**")
        surge_dept = st.selectbox("Department", dept_options, key="surge_dept")
        if st.button("🚑 Simulate surge (8 patients)"):
            dept = None if surge_dept == "(random)" else surge_dept
            batch = generate_arrival_batch(hist, n=8, force_department=dept)
            st.session_state.live_df = assign_beds(
                pd.concat([st.session_state.live_df, batch], ignore_index=True),
                config.DEPARTMENT_BED_CAPACITY,
            )
            n_crit = int(batch["is_critical"].sum())
            _log_event(f"Surge: 8 arrivals in {batch.iloc[0]['department']} ({n_crit} critical)")
            _recalculate()
            st.rerun()

    with col4:
        st.markdown("**Reset**")
        if st.button("↺ Reset simulation"):
            for key in ["initialized"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.divider()
    st.subheader("Event log")
    for e in st.session_state.event_log:
        st.text(f"[{e['time']}] {e['message']}")


# ---------------------------------------------------------------------------
# Benchmark tab
# ---------------------------------------------------------------------------

with tab_benchmark:
    st.subheader("CPU (pandas) vs. GPU (cudf.pandas) benchmark")
    st.caption(
        "This deployed instance has no GPU -- that's a real constraint of free-tier "
        "hosting, not an oversight. The CPU benchmark below runs live, right here. "
        "The GPU numbers come from the same code run in Colab on an NVIDIA T4; "
        "upload that run's results below to see both on one chart."
    )

    size_options = {"20k": 20_000, "100k": 100_000, "300k": 300_000, "1M": 1_000_000}
    chosen = st.multiselect("Dataset sizes to test", list(size_options.keys()), default=["20k", "100k", "300k"])

    if st.button("Run CPU benchmark now"):
        rows = []
        progress = st.progress(0.0, text="Running...")
        for i, label in enumerate(chosen):
            n = size_options[label]
            bench_df = generate_baseline(st.session_state.historical, n_rows=n, span_days=365, seed=99)
            t0 = time.perf_counter()
            _ = run_feature_pipeline(bench_df, config.DEPARTMENT_BED_CAPACITY)
            t1 = time.perf_counter()
            rows.append({"n_rows": n, "seconds": round(t1 - t0, 4), "backend": "pandas"})
            progress.progress((i + 1) / len(chosen), text=f"{label} done")
        st.session_state.cpu_bench = pd.DataFrame(rows)
        progress.empty()

    if "cpu_bench" in st.session_state:
        st.markdown("**CPU results (this session)**")
        st.dataframe(st.session_state.cpu_bench, width='stretch')

    st.divider()
    st.markdown("**Attempt a live GPU run** (only succeeds on a machine with an NVIDIA GPU + cuDF installed)")
    if st.button("Try GPU benchmark now"):
        n = size_options[chosen[0]] if chosen else 20_000
        bench_df = generate_baseline(st.session_state.historical, n_rows=n, span_days=365, seed=99)
        tmp_path = config.DATA_DIR / "_bench_tmp.parquet"
        bench_df.to_parquet(tmp_path)
        result = subprocess.run(
            [sys.executable, str(config.PROJECT_ROOT / "run_pipeline_backend.py"),
             "--backend", "cudf", "--input", str(tmp_path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            st.success(result.stdout)
        else:
            st.warning("GPU path unavailable in this environment (expected on free-tier hosting):")
            st.code(result.stdout or result.stderr)

    st.divider()
    st.markdown("**Upload Colab GPU benchmark results (CSV with columns: n_rows, seconds, backend)**")
    uploaded = st.file_uploader("benchmark_results.csv", type="csv")
    if uploaded is not None:
        gpu_bench = pd.read_csv(uploaded)
        combined = gpu_bench
        if "cpu_bench" in st.session_state:
            combined = pd.concat([st.session_state.cpu_bench, gpu_bench], ignore_index=True)
        pivot = combined.pivot_table(index="n_rows", columns="backend", values="seconds", aggfunc="mean").sort_index()
        st.markdown("**Combined CPU vs GPU**")
        st.bar_chart(pivot)
        if "pandas" in pivot.columns and "cudf.pandas" in pivot.columns:
            pivot["speedup_x"] = (pivot["pandas"] / pivot["cudf.pandas"]).round(1)
            st.dataframe(pivot, width='stretch')