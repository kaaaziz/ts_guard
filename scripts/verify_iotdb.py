from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



from integrations.iotdb import get_iotdb_settings, IoTDBCanonicalWriter


def main():
    settings = get_iotdb_settings()

    if not settings.enabled:
        print("IoTDB is disabled in .env")
        return

    writer = IoTDBCanonicalWriter(settings)
    writer.connect()

    print("=== SHOW DATABASES ===")
    print(writer.query_df("SHOW DATABASES"))

    print("\n=== SHOW TIMESERIES ===")
    print(writer.query_df(f"SHOW TIMESERIES {settings.database}.**"))

    print("\n=== SAMPLE QUERY ===")
    try:
        df = writer.query_df(f"SELECT ** FROM {settings.database}.** LIMIT 20")
        print(df)
    except Exception as exc:
        print(f"No rows yet or query not ready: {exc}")


if __name__ == "__main__":
    main()