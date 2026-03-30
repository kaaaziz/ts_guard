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
    print("IoTDB connection OK")
    print(f"Database ready: {settings.database}")
    print(f"Dataset node: {settings.dataset_node}")


if __name__ == "__main__":
    main()