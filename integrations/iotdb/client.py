from __future__ import annotations

from iotdb.Session import Session

from .config import IoTDBSettings


def open_session(settings: IoTDBSettings) -> Session:
    session = Session(
        settings.host,
        str(settings.port),
        settings.username,
        settings.password,
    )
    session.open(False)
    return session