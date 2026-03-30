from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import re

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_node(text: str, prefix: str) -> str:
    """
    Convert arbitrary text into a safe IoTDB path node.
    We avoid quoting complexity by forcing:
      - lowercase
      - only [a-z0-9_]
      - no leading digit
    """
    value = str(text).strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = value.strip("_")
    if not value:
        value = prefix
    if value[0].isdigit():
        value = f"{prefix}_{value}"
    return value


@dataclass(frozen=True)
class IoTDBSettings:
    enabled: bool
    host: str
    port: int
    username: str
    password: str
    database: str
    dataset_node: str
    model_version: str
    ignore_late: bool


def get_iotdb_settings() -> IoTDBSettings:
    enabled = _env_bool("TSGUARD_IOTDB_ENABLED", False)
    host = os.getenv("TSGUARD_IOTDB_HOST", "127.0.0.1").strip()
    port = int(os.getenv("TSGUARD_IOTDB_PORT", "6667"))
    username = os.getenv("TSGUARD_IOTDB_USER", "root").strip()
    password = os.getenv("TSGUARD_IOTDB_PASSWORD", "root").strip()
    database = os.getenv("TSGUARD_IOTDB_DATABASE", "root.tsguard").strip()
    dataset_raw = os.getenv("TSGUARD_IOTDB_DATASET", "pm25").strip()
    model_version = os.getenv("TSGUARD_IOTDB_MODEL_VERSION", "model_TSGuard.pth").strip()
    ignore_late = _env_bool("TSGUARD_IOTDB_IGNORE_LATE", True)

    return IoTDBSettings(
        enabled=enabled,
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        dataset_node=_sanitize_node(dataset_raw, prefix="dataset"),
        model_version=model_version,
        ignore_late=ignore_late,
    )


def make_run_id(dataset_node: str) -> str:
    """
    Example:
      pm25_run_2026_03_16_14_30_15
    """
    safe_dataset = _sanitize_node(dataset_node, prefix="dataset")
    stamp = datetime.now().astimezone().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{safe_dataset}_run_{stamp}"