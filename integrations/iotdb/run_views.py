from __future__ import annotations

from pathlib import Path
from typing import Literal
import numbers

import pandas as pd

from .config import get_iotdb_settings
from .writer import IoTDBCanonicalWriter


VIEW_MODE = Literal["A", "B"]


def _base_prefix() -> str:
    settings = get_iotdb_settings()
    return f"{settings.database}.{settings.dataset_node}"


def _sensor_id_from_node(sensor_node: str) -> str:
    sensor_node = str(sensor_node)
    if sensor_node.startswith("sensor_"):
        return sensor_node[len("sensor_"):]
    return sensor_node


def _time_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).lower() == "time":
            return col
    for col in df.columns:
        if "time" in str(col).lower():
            return col
    return df.columns[0]


def _to_datetime(value) -> pd.Timestamp:
    """
    Parse IoTDB query timestamps correctly.

    In our setup, IoTDB uses millisecond precision, so numeric time values
    returned by the query layer must be interpreted as epoch milliseconds.
    """
    if value is None or pd.isna(value):
        return pd.NaT

    # Numeric values coming back from IoTDB should be treated as epoch milliseconds
    if isinstance(value, numbers.Integral):
        return pd.to_datetime(int(value), unit="ms", errors="coerce")

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return pd.to_datetime(int(value), unit="ms", errors="coerce")

    # Numeric strings should also be treated as epoch milliseconds
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return pd.NaT
        if s.lstrip("-").isdigit():
            return pd.to_datetime(int(s), unit="ms", errors="coerce")
        return pd.to_datetime(s, errors="coerce")

    # Fall back to pandas for already formatted datetime-like objects
    return pd.to_datetime(value, errors="coerce")


def _iter_path_strings(df: pd.DataFrame):
    base = _base_prefix() + "."
    for col in df.columns:
        series = df[col]
        for value in series:
            if isinstance(value, str) and value.startswith(base):
                yield value.strip()


def list_run_ids(writer: IoTDBCanonicalWriter) -> list[str]:
    base = _base_prefix()
    try:
        df = writer.query_df(f"SHOW TIMESERIES {base}.**")
    except Exception:
        return []

    run_ids = set()
    base_len = len(base.split("."))

    for path in _iter_path_strings(df):
        parts = path.split(".")
        if len(parts) >= base_len + 3:
            run_ids.add(parts[base_len])

    return sorted(run_ids)


def load_run_long_df(writer: IoTDBCanonicalWriter, run_id: str) -> pd.DataFrame:
    """
    Returns one row per (datetime, sensor_id), with columns:
      datetime, sensor_id, value, source_kind, constraint_flag, strategy, model_version
    """
    settings = get_iotdb_settings()
    run_root = f"{settings.database}.{settings.dataset_node}.{run_id}"

    try:
        raw = writer.query_df(f"SELECT ** FROM {run_root}.**")
    except Exception:
        return pd.DataFrame(
            columns=[
                "datetime",
                "sensor_id",
                "value",
                "source_kind",
                "constraint_flag",
                "strategy",
                "model_version",
            ]
        )

    if raw.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "sensor_id",
                "value",
                "source_kind",
                "constraint_flag",
                "strategy",
                "model_version",
            ]
        )

    time_col = _time_column(raw)
    base_len = len(run_root.split("."))
    records = []

    for _, row in raw.iterrows():
        ts = _to_datetime(row[time_col])
        if pd.isna(ts):
            continue

        per_sensor: dict[str, dict] = {}

        for col in raw.columns:
            if col == time_col:
                continue

            value = row[col]
            if pd.isna(value):
                continue

            col_name = str(col)
            parts = col_name.split(".")

            if len(parts) < base_len + 2:
                continue

            sensor_node = parts[base_len]
            measurement = parts[base_len + 1]
            sensor_id = _sensor_id_from_node(sensor_node)

            bucket = per_sensor.setdefault(
                sensor_id,
                {
                    "datetime": ts,
                    "sensor_id": sensor_id,
                    "value": None,
                    "source_kind": None,
                    "constraint_flag": None,
                    "strategy": None,
                    "model_version": None,
                },
            )
            bucket[measurement] = value

        records.extend(per_sensor.values())

    long_df = pd.DataFrame(records)
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "sensor_id",
                "value",
                "source_kind",
                "constraint_flag",
                "strategy",
                "model_version",
            ]
        )

    long_df["datetime"] = pd.to_datetime(long_df["datetime"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["source_kind"] = long_df["source_kind"].astype("string")
    long_df["strategy"] = long_df["strategy"].astype("string")
    long_df["model_version"] = long_df["model_version"].astype("string")

    def _to_bool(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in {"true", "1"}:
            return True
        if s in {"false", "0"}:
            return False
        return pd.NA

    long_df["constraint_flag"] = long_df["constraint_flag"].map(_to_bool)
    long_df = long_df.sort_values(["datetime", "sensor_id"]).reset_index(drop=True)
    return long_df


def build_wide_view(long_df: pd.DataFrame, mode: VIEW_MODE) -> pd.DataFrame:
    """
    Mode A:
      only imputed/fallback values are shown; real values stay empty
    Mode B:
      final canonical values (real + imputed + fallback)
    """
    if long_df.empty:
        return pd.DataFrame(columns=["datetime"])

    work = long_df.copy()

    if mode == "A":
        work = work[work["source_kind"].isin(["imputed", "fallback"])].copy()
    elif mode == "B":
        pass
    else:
        raise ValueError("mode must be 'A' or 'B'")

    if work.empty:
        return pd.DataFrame(columns=["datetime"])

    wide = (
        work.pivot_table(
            index="datetime",
            columns="sensor_id",
            values="value",
            aggfunc="last",
        )
        .sort_index()
        .sort_index(axis=1)
        .reset_index()
    )
    return wide


def list_run_summaries(writer: IoTDBCanonicalWriter) -> pd.DataFrame:
    rows = []
    for run_id in list_run_ids(writer):
        long_df = load_run_long_df(writer, run_id)
        if long_df.empty:
            rows.append(
                {
                    "run_id": run_id,
                    "rows": 0,
                    "sensors": 0,
                    "start": pd.NaT,
                    "end": pd.NaT,
                }
            )
            continue

        rows.append(
            {
                "run_id": run_id,
                "rows": int(len(long_df)),
                "sensors": int(long_df["sensor_id"].nunique()),
                "start": long_df["datetime"].min(),
                "end": long_df["datetime"].max(),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["run_id", "rows", "sensors", "start", "end"])

    return out.sort_values("run_id").reset_index(drop=True)


def build_run_view(writer: IoTDBCanonicalWriter, run_id: str, mode: VIEW_MODE) -> pd.DataFrame:
    long_df = load_run_long_df(writer, run_id)
    return build_wide_view(long_df, mode)


def export_run_view(df: pd.DataFrame, run_id: str, mode: VIEW_MODE) -> Path:
    out_dir = Path("outputs/run_views")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "imputed_only" if mode == "A" else "canonical_all"
    out_path = out_dir / f"{run_id}__{suffix}.csv"
    df.to_csv(out_path, index=False)
    return out_path