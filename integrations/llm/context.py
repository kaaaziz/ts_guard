from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re

import numpy as np
import pandas as pd
import streamlit as st

from utils.config import DEFAULT_VALUES


@dataclass
class SystemSnapshot:
    running: bool
    training: bool
    page: str
    sim_ptr: int
    sim_total: int
    current_ts: Optional[pd.Timestamp]
    missing_now: Optional[int]
    imputed_now: Optional[int]
    worst_missing_sensors: List[str]
    global_missing_pct: Optional[float]
    constraint_sensitivity: Optional[float]
    recent_neighbor_alerts: List[str]
    recent_constraint_alerts: List[str]
    sim_seconds_per_hour: Optional[float]
    sigma_minutes: Optional[float]
    missing_thresholds: Optional[Dict[str, tuple]]
    baseline_sensor_count: Optional[int]
    dynamic_captor_count: Optional[int]


def _find_current_ts(ss) -> Optional[pd.Timestamp]:
    base = ss.get("orig_missing_baseline")
    if not isinstance(base, pd.DataFrame) or base.index.empty:
        return None

    ptr = int(ss.get("sim_ptr", 0))
    if ptr <= 0:
        idx = 0
    else:
        idx = min(ptr - 1, len(base.index) - 1)

    try:
        return pd.Timestamp(base.index[idx])
    except Exception:
        return None


def build_snapshot() -> SystemSnapshot:
    ss = st.session_state

    running = bool(ss.get("running", False))
    training = bool(ss.get("training", False))
    page = str(ss.get("page", "sim"))
    sim_ptr = int(ss.get("sim_ptr", 0))
    sim_total = int(ss.get("sim_total_steps", 0))

    current_ts = _find_current_ts(ss)

    missing_now = None
    worst_sensors: List[str] = []
    global_pct: Optional[float] = None

    base = ss.get("orig_missing_baseline")
    if isinstance(base, pd.DataFrame) and not base.empty:
        if current_ts is not None and current_ts in base.index:
            row = base.loc[current_ts]
            missing_now = int(row.sum())
            worst_sensors = [c for c, v in row.items() if bool(v)][:8]

        total_missing = int(np.nansum(base.values))
        total_cells = int(base.size)
        if total_cells > 0:
            global_pct = 100.0 * total_missing / total_cells

    imputed_now = None
    imp_mask = ss.get("imputed_mask")
    if (
        current_ts is not None
        and isinstance(imp_mask, pd.DataFrame)
        and not imp_mask.index.empty
    ):
        try:
            row_imp = imp_mask.reindex(
                index=[current_ts], columns=imp_mask.columns, fill_value=False
            ).iloc[0]
            imputed_now = int(row_imp.sum())
        except Exception:
            imputed_now = None

    constraint_sensitivity = ss.get("constraint_sensitivity", None)

    recent_neighbor_alerts: List[str] = []
    recent_constraint_alerts: List[str] = []

    alerts = ss.get("_grp_alerts", {})
    if isinstance(alerts, dict):
        groups = list(alerts.values())
        groups.sort(key=lambda g: g.get("ts", pd.Timestamp.min), reverse=True)
        for g in groups:
            if g.get("dismissed"):
                continue
            title = str(g.get("title", ""))
            for item in g.get("items", []):
                txt = str(item.get("text", ""))
                low = txt.lower()
                if "neighbor" in low or "neighbour" in low or "spatial" in title.lower():
                    recent_neighbor_alerts.append(txt)
                if "constraint" in low or "domain" in low or "range" in low:
                    recent_constraint_alerts.append(txt)
            if len(recent_neighbor_alerts) >= 5 and len(recent_constraint_alerts) >= 5:
                break

    sim_seconds_per_hour = ss.get("sim_seconds_per_hour", None)
    try:
        if sim_seconds_per_hour is not None:
            sim_seconds_per_hour = float(sim_seconds_per_hour)
    except (TypeError, ValueError):
        sim_seconds_per_hour = None

    sigma_minutes = ss.get("sigma_threshold", None)
    try:
        if sigma_minutes is not None:
            sigma_minutes = float(sigma_minutes)
    except (TypeError, ValueError):
        sigma_minutes = None

    missing_thresholds = ss.get("missing_value_thresholds")
    if not missing_thresholds:
        try:
            missing_thresholds = {
                "Green": (
                    DEFAULT_VALUES["gauge_green_min"],
                    DEFAULT_VALUES["gauge_green_max"],
                ),
                "Yellow": (
                    DEFAULT_VALUES["gauge_yellow_min"],
                    DEFAULT_VALUES["gauge_yellow_max"],
                ),
                "Red": (
                    DEFAULT_VALUES["gauge_red_min"],
                    DEFAULT_VALUES["gauge_red_max"],
                ),
            }
        except Exception:
            missing_thresholds = None

    baseline_sensor_count = None
    if isinstance(base, pd.DataFrame):
        baseline_sensor_count = len(base.columns)

    dynamic_captor_count = None
    dyn = ss.get("dynamic_captors")
    if isinstance(dyn, dict):
        dynamic_captor_count = len(dyn)

    return SystemSnapshot(
        running=running,
        training=training,
        page=page,
        sim_ptr=sim_ptr,
        sim_total=sim_total,
        current_ts=current_ts,
        missing_now=missing_now,
        imputed_now=imputed_now,
        worst_missing_sensors=worst_sensors,
        global_missing_pct=global_pct,
        constraint_sensitivity=constraint_sensitivity,
        recent_neighbor_alerts=recent_neighbor_alerts,
        recent_constraint_alerts=recent_constraint_alerts,
        sim_seconds_per_hour=sim_seconds_per_hour,
        sigma_minutes=sigma_minutes,
        missing_thresholds=missing_thresholds,
        baseline_sensor_count=baseline_sensor_count,
        dynamic_captor_count=dynamic_captor_count,
    )


def _resolve_sensor_column(raw_id: str) -> Optional[str]:
    ss = st.session_state
    candidates: set[str] = set()

    base = ss.get("orig_missing_baseline")
    if isinstance(base, pd.DataFrame):
        candidates.update(map(str, base.columns))

    imp = ss.get("imputed_mask")
    if isinstance(imp, pd.DataFrame):
        candidates.update(map(str, imp.columns))

    if not candidates:
        return None

    raw = str(raw_id).strip()
    raw_core = re.sub(r"[^0-9A-Za-z_-]", "", raw)

    possible_keys = []
    possible_keys.append(raw_core)

    core_no0 = raw_core.lstrip("0") or "0"
    possible_keys.append(core_no0)

    if core_no0.isdigit():
        possible_keys.append(core_no0.zfill(6))

    lower_map = {c.lower(): c for c in candidates}
    for key in possible_keys:
        if key.lower() in lower_map:
            return lower_map[key.lower()]

    for key in possible_keys:
        k = key.lower()
        for cand_lower, cand_orig in lower_map.items():
            if k and k in cand_lower:
                return cand_orig

    return None


def _extract_sensor_reference(user_text: str) -> Optional[str]:
    m = re.search(
        r"(sensor|captor|station)\s+([0-9A-Za-z_-]+)",
        user_text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(2)
    return None


def build_sensor_detail_reply(raw_id: str, snap: SystemSnapshot) -> str:
    ss = st.session_state
    col = _resolve_sensor_column(raw_id)
    if col is None:
        examples: list[str] = []
        base = ss.get("orig_missing_baseline")
        if isinstance(base, pd.DataFrame):
            examples = [str(c) for c in list(base.columns)[:6]]
        elif isinstance(ss.get("imputed_mask"), pd.DataFrame):
            im = ss["imputed_mask"]
            examples = [str(c) for c in list(im.columns)[:6]]

        msg = f"I couldn't find any sensor / captor matching **{raw_id}**."
        if examples:
            msg += " Try using an exact ID such as: `" + "`, `".join(examples) + "`."
        return msg

    base = ss.get("orig_missing_baseline")
    im_mask = ss.get("imputed_mask")
    global_df = ss.get("global_df")

    is_static = isinstance(base, pd.DataFrame) and col in base.columns

    ptr = int(ss.get("sim_ptr", 0))
    missing_pct = None
    imputed_pct = None

    if is_static and isinstance(base, pd.DataFrame) and not base.index.empty:
        idx_upto = base.index[: min(ptr, len(base.index))] if ptr > 0 else base.index[:0]
        if len(idx_upto) > 0:
            col_mask = base.loc[idx_upto, col].astype(bool)
            n_total = len(col_mask)
            n_missing = int(col_mask.sum())
            if n_total > 0:
                missing_pct = 100.0 * n_missing / n_total

    if isinstance(im_mask, pd.DataFrame) and not im_mask.index.empty and col in im_mask.columns:
        idx_upto = im_mask.index[: min(ptr, len(im_mask.index))] if ptr > 0 else im_mask.index[:0]
        if len(idx_upto) > 0:
            col_imp = im_mask.loc[idx_upto, col].astype(bool)
            n_total = len(col_imp)
            n_imp = int(col_imp.sum())
            if n_total > 0:
                imputed_pct = 100.0 * n_imp / n_total

    hist_line = None
    if isinstance(global_df, pd.DataFrame) and not global_df.empty and col in global_df.columns:
        if "datetime" in global_df.columns:
            series = pd.to_numeric(global_df[col], errors="coerce")
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if not series.empty:
                n_hist = len(series)
                hist_line = (
                    f"- Values seen so far: min **{series.min():.2f}**, "
                    f"median **{series.median():.2f}**, "
                    f"max **{series.max():.2f}** over **{n_hist}** timestamps."
                )

    lines: List[str] = []
    lines.append(f"Details for sensor / captor **{col}**:")

    if is_static:
        lines.append("- Type: baseline sensor from the original dataset.")
    else:
        lines.append("- Type: dynamic captor added during this simulation run.")

    if (
        snap.current_ts is not None
        and isinstance(global_df, pd.DataFrame)
        and "datetime" in global_df.columns
        and col in global_df.columns
    ):
        ts = snap.current_ts
        ts_str = ts.strftime("%Y-%m-%d %H:%M")
        rows = global_df[global_df["datetime"] == ts]
        if not rows.empty:
            v = rows.iloc[-1][col]
            if pd.notna(v):
                try:
                    v_f = float(v)
                    v_str = f"{v_f:.2f}"
                except Exception:
                    v_str = str(v)

                im_flag = False
                if isinstance(im_mask, pd.DataFrame) and ts in im_mask.index and col in im_mask.columns:
                    im_flag = bool(im_mask.at[ts, col])

                status = "imputed" if im_flag else "real"
                lines.append(
                    f"- Current value at **{ts_str}**: **{v_str}** ({status})."
                )
            else:
                lines.append(
                    f"- At the current simulated time **{ts_str}** I don't have a numeric value for this captor."
                )
        else:
            lines.append(
                "- I don't yet have a stored value for this captor at the current simulation time."
            )

    if missing_pct is not None:
        lines.append(
            f"- Originally missing readings so far: **{missing_pct:.1f}%** "
            "of expected timestamps."
        )
    elif is_static:
        lines.append(
            "- I can't yet compute original-missing statistics for this sensor."
        )

    if imputed_pct is not None:
        if is_static:
            lines.append(
                f"- Fraction of timestamps where TSGuard imputed this sensor: "
                f"**{imputed_pct:.1f}%** of processed steps."
            )
        else:
            lines.append(
                f"- Fraction of processed steps where TSGuard produced a value "
                f"for this dynamic captor: **{imputed_pct:.1f}%**."
            )

    if hist_line:
        lines.append(hist_line)

    if len(lines) == 1:
        lines.append(
            "I don't have enough information about this captor yet. "
            "Run the simulation for a bit longer and ask again."
        )

    return "\n".join(lines)


def _build_status_reply(snap: SystemSnapshot) -> str:
    if snap.running:
        ts_str = (
            snap.current_ts.strftime("%Y-%m-%d %H:%M")
            if snap.current_ts is not None
            else "n/a"
        )

        lines: List[str] = []
        lines.append(
            "The simulation is currently **running** on the TSGuard simulation page."
        )
        lines.append("")
        lines.append(f"- Current simulated time: **{ts_str}**")

        if snap.sim_ptr > 0:
            lines.append(
                f"- Processed timestamps so far in this run: **{snap.sim_ptr}**."
            )
        else:
            lines.append(
                "- The replay just started; no timestamps have been processed yet."
            )

        return "\n".join(lines)

    if snap.training:
        return (
            "Model **training** is running right now. "
            "The simulation is paused until training finishes."
        )

    return (
        "The simulation is currently **stopped**. "
        "Use **Start TSGuard Simulation** to sweep through the data."
    )


def _build_missing_reply(snap: SystemSnapshot) -> str:
    if snap.current_ts is None:
        return (
            "I don't have a current simulation timestamp yet. "
            "Start the simulation and I'll summarise missing values at each step."
        )

    ts_str = snap.current_ts.strftime("%Y-%m-%d %H:%M")
    lines = [f"At simulated time **{ts_str}**:"]

    if snap.missing_now is not None:
        lines.append(
            f"- Sensors with delayed / missing measurements: **{snap.missing_now}**."
        )
    if snap.imputed_now is not None:
        lines.append(
            f"- Sensors currently using imputed values: **{snap.imputed_now}**."
        )
    if snap.worst_missing_sensors:
        lines.append(
            "- Examples of affected sensors: `"
            + ", ".join(snap.worst_missing_sensors[:5])
            + "`."
        )
    if snap.global_missing_pct is not None:
        lines.append(
            f"- So far, about **{snap.global_missing_pct:.1f}%** of expected readings "
            "were originally missing."
        )
    if len(lines) == 1:
        lines.append("I don't see any missing values at this exact timestamp.")
    return "\n".join(lines)


def _build_constraint_reply(snap: SystemSnapshot) -> str:
    if not snap.recent_neighbor_alerts and not snap.recent_constraint_alerts:
        base = "Right now I don't see any active spatial or constraint alerts."
        if snap.constraint_sensitivity is not None:
            base += (
                f" Constraint sensitivity is **{snap.constraint_sensitivity:.2f}** "
                "(0 = only large violations, 1 = very sensitive)."
            )
        return base

    lines: List[str] = []

    if snap.recent_neighbor_alerts:
        lines.append("Recent **neighbor / spatial anomaly** alerts:")
        for msg in snap.recent_neighbor_alerts[:3]:
            lines.append(f"- {msg}")

    if snap.recent_constraint_alerts:
        lines.append("\nRecent **constraint violations**:")
        for msg in snap.recent_constraint_alerts[:3]:
            lines.append(f"- {msg}")

    if snap.constraint_sensitivity is not None:
        lines.append(
            f"\nThe constraint sensitivity slider is currently at "
            f"**{snap.constraint_sensitivity:.2f}** "
            "(0 = only large violations, 1 = very sensitive)."
        )
    return "\n".join(lines)


def _build_simulation_config_reply(snap: SystemSnapshot) -> str:
    tau = snap.sim_seconds_per_hour

    if tau is None:
        return (
            "I can't see the simulation speed slider value yet. "
            "Open **Settings → Simulation** to configure it."
        )

    if abs(tau) < 1e-9:
        lines = [
            "The simulation speed slider is at **0.0**.",
            "That means TSGuard runs at **max speed**, with no intentional pause "
            "between timestamps.",
        ]
    else:
        hours_per_sec = 1.0 / tau
        lines = [
            f"The simulation is configured so that **1 simulated hour ≈ {tau:.2f} real seconds**.",
            f"Equivalently, TSGuard replays about **{hours_per_sec:.2f} simulated hours per real second**.",
        ]

    lines.append(
        "You can adjust this in **Settings → Simulation**; changes apply to subsequent timestamps."
    )
    return "\n".join(lines)


def _build_thresholds_reply(snap: SystemSnapshot) -> str:
    lines: List[str] = []

    if snap.sigma_minutes is not None:
        minutes = snap.sigma_minutes
        hours = minutes / 60.0
        if abs(minutes - round(minutes)) < 1e-6:
            minutes_txt = f"{minutes:.0f}"
        else:
            minutes_txt = f"{minutes:.1f}"
        lines.append(
            f"The current **delay threshold σ** is **{minutes_txt} minutes** "
            f"(≈ {hours:.2f} hours)."
        )
    else:
        lines.append(
            "I can't see a stored value for the **delay threshold σ** yet. "
            "Set it in **Settings → Threshold**."
        )

    thr = snap.missing_thresholds
    if isinstance(thr, dict):
        def band(name: str) -> Optional[str]:
            vals = thr.get(name)
            if not vals or len(vals) != 2:
                return None
            lo, hi = vals
            return f"{lo}–{hi}%"

        green = band("Green")
        yellow = band("Yellow")
        red = band("Red")

        sublines = []
        if green:
            sublines.append(f"- **Green**: {green}")
        if yellow:
            sublines.append(f"- **Yellow**: {yellow}")
        if red:
            sublines.append(f"- **Red**: {red}")

        if sublines:
            lines.append("Missing-data gauge zones are currently defined as:")
            lines.extend(sublines)

        if snap.global_missing_pct is not None:
            pct = snap.global_missing_pct
            zone = None
            for name, (lo, hi) in thr.items():
                try:
                    if lo <= pct <= hi:
                        zone = name
                        break
                except Exception:
                    continue

            if zone:
                lines.append(
                    f"Across the processed history so far, about **{pct:.1f}%** "
                    f"of readings were originally missing — this falls in the "
                    f"**{zone}** zone."
                )
            else:
                lines.append(
                    f"Across the processed history so far, about **{pct:.1f}%** "
                    "of readings were originally missing (outside the configured gauge bands)."
                )
    else:
        lines.append(
            "Gauge zones for missing data haven't been saved yet. "
            "Use **Settings → Missing values** to set Green/Yellow/Red ranges."
        )

    return "\n".join(lines)


def _build_captor_reply(snap: SystemSnapshot) -> str:
    lines: List[str] = []

    if snap.baseline_sensor_count is not None:
        lines.append(
            f"I currently see **{snap.baseline_sensor_count} baseline sensors** "
            "in the dataset."
        )
    else:
        lines.append(
            "I can't see the baseline sensor table yet, so I don't know how many sensors are configured."
        )

    if snap.dynamic_captor_count is not None and snap.dynamic_captor_count > 0:
        lines.append(
            f"During this run you've added **{snap.dynamic_captor_count} dynamic captor(s)** "
            "via the *Dynamic Captors* panel. They are immediately included in the map and "
            "in the spatial imputer."
        )
    else:
        lines.append(
            "I don't see any extra dynamic captors added in this session yet."
        )

    if snap.current_ts is not None:
        ts_str = snap.current_ts.strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"All active captors report (or are imputed) at the current simulated time **{ts_str}**."
        )

    return "\n".join(lines)


def generate_rule_based_reply(user_text: str, snap: SystemSnapshot) -> str:
    text = user_text.strip()
    if not text:
        return (
            "I didn't catch anything. You can ask, for example:\n"
            "- \"What is the current status?\"\n"
            "- \"Which values are missing right now?\"\n"
            "- \"Any neighbour stations behaving abnormally?\""
        )

    lower = text.lower()
    pieces: List[str] = []

    if any(w in lower for w in ("hello", "hi ", " hi", "hey", "bonjour", "salut")):
        pieces.append(
            "Hello, I'm the **TSGuard assistant**. "
            "I can explain what's happening in the simulation, missing values, "
            "and constraint alerts."
        )

    if any(w in lower for w in ("status", "state", "running", "progress", "simulation")):
        pieces.append(_build_status_reply(snap))

    if any(
        w in lower
        for w in ("speed", "fast", "slow", "seconds per hour", "real time", "real-time")
    ):
        pieces.append(_build_simulation_config_reply(snap))

    if any(
        w in lower
        for w in ("missing", "delay", "delayed", "late", "gap", "imputed", "imputation")
    ):
        pieces.append(_build_missing_reply(snap))

    if any(
        w in lower
        for w in (
            "threshold",
            "sigma",
            "gauge",
            "green zone",
            "yellow zone",
            "red zone",
            "severity",
            "risk",
        )
    ):
        pieces.append(_build_thresholds_reply(snap))

    if any(
        w in lower
        for w in (
            "neighbor",
            "neighbour",
            "spatial",
            "constraint",
            "alert",
            "warning",
            "anomaly",
        )
    ):
        pieces.append(_build_constraint_reply(snap))

    if (
        ("sensor" in lower or "captor" in lower or "station" in lower)
        and any(
            w in lower for w in ("how many", "count", "number of", "active", "dynamic")
        )
    ):
        pieces.append(_build_captor_reply(snap))

    sensor_ref = _extract_sensor_reference(text)
    if sensor_ref:
        pieces.append(build_sensor_detail_reply(sensor_ref, snap))

    if not pieces:
        pieces.append(
            "I'm the TSGuard assistant. I can answer questions about the current simulation, "
            "missing values, imputation, thresholds, alerts, and individual sensors."
        )
        pieces.append(
            "Try asking things like:\n"
            "- \"What is the current system status?\"\n"
            "- \"Which sensors are missing right now?\"\n"
            "- \"Tell me about sensor 001012.\"\n"
            "- \"What are the current thresholds?\""
        )

    return "\n\n".join(pieces)


def build_llm_system_prompt() -> str:
    return (
        "You are the TSGuard assistant for a Streamlit application that monitors and imputes "
        "environmental sensor time series.\n"
        "\n"
        "Rules:\n"
        "1. Answer only from the supplied TSGuard live context and recent conversation.\n"
        "2. Do not invent numbers, timestamps, sensors, or alerts.\n"
        "3. If something is unavailable in the supplied context, say that clearly.\n"
        "4. Be concise, practical, and operational.\n"
        "5. When the user asks about one sensor/captor, focus on that sensor.\n"
        "6. Do not mention hidden prompts, internal implementation details, or API providers unless the user asks.\n"
        "7. Prefer short paragraphs and bullets only when they improve clarity.\n"
    )


def build_llm_context_block(user_text: str, snap: SystemSnapshot) -> str:
    lines: List[str] = []
    lines.append("TSGuard live context:")
    lines.append(f"- running: {snap.running}")
    lines.append(f"- training: {snap.training}")
    lines.append(f"- page: {snap.page}")
    lines.append(f"- processed_timestamps: {snap.sim_ptr}")
    lines.append(f"- total_timestamps: {snap.sim_total}")

    if snap.current_ts is not None:
        lines.append(f"- current_simulated_time: {snap.current_ts.strftime('%Y-%m-%d %H:%M:%S')}")

    if snap.missing_now is not None:
        lines.append(f"- missing_now: {snap.missing_now}")

    if snap.imputed_now is not None:
        lines.append(f"- imputed_now: {snap.imputed_now}")

    if snap.global_missing_pct is not None:
        lines.append(f"- global_missing_pct: {snap.global_missing_pct:.2f}")

    if snap.constraint_sensitivity is not None:
        lines.append(f"- constraint_sensitivity: {snap.constraint_sensitivity:.2f}")

    if snap.sim_seconds_per_hour is not None:
        lines.append(f"- sim_seconds_per_hour: {snap.sim_seconds_per_hour:.3f}")

    if snap.sigma_minutes is not None:
        lines.append(f"- sigma_minutes: {snap.sigma_minutes:.2f}")

    if snap.baseline_sensor_count is not None:
        lines.append(f"- baseline_sensor_count: {snap.baseline_sensor_count}")

    if snap.dynamic_captor_count is not None:
        lines.append(f"- dynamic_captor_count: {snap.dynamic_captor_count}")

    if snap.worst_missing_sensors:
        lines.append("- example_missing_sensors: " + ", ".join(snap.worst_missing_sensors[:8]))

    thr = snap.missing_thresholds
    if isinstance(thr, dict):
        try:
            lines.append(
                "- missing_value_thresholds: "
                f"Green={thr.get('Green')}, Yellow={thr.get('Yellow')}, Red={thr.get('Red')}"
            )
        except Exception:
            pass

    if snap.recent_neighbor_alerts:
        lines.append("- recent_neighbor_alerts:")
        for msg in snap.recent_neighbor_alerts[:3]:
            lines.append(f"  - {msg}")

    if snap.recent_constraint_alerts:
        lines.append("- recent_constraint_alerts:")
        for msg in snap.recent_constraint_alerts[:3]:
            lines.append(f"  - {msg}")

    sensor_ref = _extract_sensor_reference(user_text)
    if sensor_ref:
        detail = build_sensor_detail_reply(sensor_ref, snap)
        lines.append("")
        lines.append("Relevant sensor detail:")
        lines.extend([f"- {ln}" for ln in detail.splitlines() if ln.strip()])

    return "\n".join(lines)