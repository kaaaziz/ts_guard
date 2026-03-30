from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from integrations.iotdb import get_iotdb_settings, IoTDBCanonicalWriter
from integrations.iotdb.run_views import (
    list_run_summaries,
    build_run_view,
    export_run_view,
)


def _prompt_run_index(max_index: int) -> int:
    while True:
        raw = input(f"\nChoose run number [1-{max_index}]: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= max_index:
                return idx - 1
        except Exception:
            pass
        print("Invalid selection. Try again.")


def _prompt_mode() -> str:
    while True:
        raw = input("\nChoose view mode [A (Show only imputed values) /B (Show full canonical dataset)]: ").strip().upper()
        if raw in {"A", "B"}:
            return raw
        print("Invalid mode. Type A or B.")


def main():
    settings = get_iotdb_settings()

    if not settings.enabled:
        print("IoTDB is disabled in .env")
        return

    writer = IoTDBCanonicalWriter(settings)
    writer.connect()

    summary = list_run_summaries(writer)
    if summary.empty:
        print("No stored runs found in IoTDB.")
        return

    print("\nAvailable runs:\n")
    display = summary.copy()
    display.index = display.index + 1
    print(display.to_string())

    idx = _prompt_run_index(len(summary))
    run_id = str(summary.iloc[idx]["run_id"])

    mode = _prompt_mode()
    df = build_run_view(writer, run_id, mode)
    out_path = export_run_view(df, run_id, mode)

    print("\n========================================")
    print(f"Selected run: {run_id}")
    print(f"View mode   : {mode}")
    print(f"Saved CSV   : {out_path}")
    print("========================================\n")

    if df.empty:
        print("No rows found for that run/view.")
        return

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 5000,
        "display.max_colwidth", None,
    ):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()