from __future__ import annotations

from typing import Iterable
import math

import pandas as pd

from .client import open_session
from .config import IoTDBSettings
from .schema import ensure_database, ensure_sensor_schema, sensor_device_path


def _sql_text(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


class IoTDBCanonicalWriter:
    """
    Canonical-output writer for TSGuard.

    Policy:
    - writes only final accepted outputs
    - ignores NaN values
    - ignores repeated (run_id, sensor_id, timestamp) writes inside the current app runtime
      when ignore_late=True
    """

    def __init__(self, settings: IoTDBSettings):
        self.settings = settings
        self.session = None
        self.known_sensors: set[str] = set()
        self.finalized_keys: set[tuple[str, int]] = set()

    def connect(self) -> None:
        if not self.settings.enabled:
            return
        if self.session is not None:
            return
        self.session = open_session(self.settings)
        ensure_database(self.session, self.settings)

    def close(self) -> None:
        if self.session is not None:
            try:
                self.session.close()
            finally:
                self.session = None

    def bootstrap(self, sensor_ids: Iterable[str], run_id: str) -> None:
        if not self.settings.enabled:
            return
        self.connect()
        for sensor_id in sensor_ids:
            self._ensure_sensor(run_id, sensor_id)

    def _ensure_sensor(self, run_id: str, sensor_id: str) -> None:
        sensor_id = str(sensor_id)
        key = (run_id, sensor_id)
        if key in self.known_sensors:
            return
        ensure_sensor_schema(self.session, self.settings, run_id, sensor_id)
        self.known_sensors.add(key)

    def write_point(
        self,
        timestamp,
        run_id: str,
        sensor_id: str,
        value: float,
        source_kind: str,
        constraint_flag: bool,
        strategy: str,
        model_version: str | None = None,
    ) -> bool:
        if not self.settings.enabled:
            return False

        if value is None:
            return False

        try:
            value = float(value)
        except Exception:
            return False

        if not math.isfinite(value):
            return False

        ts = pd.Timestamp(timestamp)
        ts_ms = int(ts.value // 10**6)

        sensor_id = str(sensor_id)
        key = (run_id, sensor_id, ts_ms)

        if self.settings.ignore_late and key in self.finalized_keys:
            return False

        self.connect()
        self._ensure_sensor(run_id, sensor_id)

        device = sensor_device_path(self.settings, run_id, sensor_id)
        model_version = model_version or self.settings.model_version

        sql = (
            f"INSERT INTO {device}"
            f"(time, value, source_kind, constraint_flag, strategy, model_version) "
            f"ALIGNED VALUES("
            f"{ts_ms}, "
            f"{value}, "
            f"{_sql_text(source_kind)}, "
            f"{str(bool(constraint_flag)).lower()}, "
            f"{_sql_text(strategy)}, "
            f"{_sql_text(model_version)}"
            f")"
        )

        self.session.execute_non_query_statement(sql)
        self.finalized_keys.add(key)
        return True

    def query_df(self, sql: str):
        self.connect()
        result = self.session.execute_query_statement(sql)
        return result.todf()