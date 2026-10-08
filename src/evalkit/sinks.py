"""Where traces go. `LocalSink` writes files; add others by subclassing `Sink`."""
from __future__ import annotations

import json
from pathlib import Path


class Sink:
    def write(self, trace: dict) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        pass


class LocalSink(Sink):
    """<dir>/<agent>/<YYYY-MM-DD>/<id>.json, one file per trace."""

    def __init__(self, directory: str | Path = ".traces"):
        self.dir = Path(directory)

    def write(self, trace: dict) -> None:
        day = trace["timestamp"][:10]
        d = self.dir / trace["agent"] / day
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{trace['id']}.json").write_text(json.dumps(trace, indent=2, default=str))
