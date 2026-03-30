from __future__ import annotations

import re

from .config import IoTDBSettings


def _sanitize_sensor_node(sensor_id: str) -> str:
    value = str(sensor_id).strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = value.strip("_")
    if not value:
        value = "unknown_sensor"
    return f"sensor_{value}"


def _sanitize_run_node(run_id: str) -> str:
    value = str(run_id).strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = value.strip("_")
    if not value:
        value = "run_unknown"
    if value[0].isdigit():
        value = f"run_{value}"
    return value


def run_root_path(settings: IoTDBSettings, run_id: str) -> str:
    return f"{settings.database}.{settings.dataset_node}.{_sanitize_run_node(run_id)}"


def sensor_device_path(settings: IoTDBSettings, run_id: str, sensor_id: str) -> str:
    return f"{run_root_path(settings, run_id)}.{_sanitize_sensor_node(sensor_id)}"


def ensure_database(session, settings: IoTDBSettings) -> None:
    try:
        session.execute_non_query_statement(f"CREATE DATABASE {settings.database}")
    except Exception as exc:
        msg = str(exc).lower()
        # Idempotent behavior for reruns
        if "already been created as database" in msg or ("already" in msg and "database" in msg):
            return
        raise


def ensure_sensor_schema(session, settings: IoTDBSettings, run_id: str, sensor_id: str) -> None:
    device = sensor_device_path(settings, run_id, sensor_id)
    sql = (
        f"CREATE ALIGNED TIMESERIES {device} ("
        f"value DOUBLE, "
        f"source_kind TEXT, "
        f"constraint_flag BOOLEAN, "
        f"strategy TEXT, "
        f"model_version TEXT"
        f")"
    )
    try:
        session.execute_non_query_statement(sql)
    except Exception as exc:
        msg = str(exc).lower()
        if "already" in msg and "exist" in msg:
            return
        raise