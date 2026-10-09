from __future__ import annotations

import errno
import json
import os
from contextlib import contextmanager
from pathlib import Path


class CheckpointError(ValueError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path: Path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CheckpointError(f"Duplicate JSON key in {path}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise CheckpointError(f"Non-finite JSON value in {path}: {value}")

    try:
        return json.loads(path.read_bytes(), object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise CheckpointError(f"Invalid JSON in {path}; preserve files and inspect them before retrying") from exc


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(fd)


def durable_directory(path: Path, mode: int = 0o700) -> None:
    if path.is_dir():
        return
    durable_directory(path.parent, mode)
    try:
        path.mkdir(mode=mode)
    except FileExistsError:
        if not path.is_dir():
            raise
    sync_directory(path.parent)


def promote_temp(temp: Path, path: Path) -> None:
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)
    sync_directory(path.parent)


def atomic_write(path: Path, value) -> None:
    payload = canonical(value)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("wb") as handle:
            os.chmod(temp, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        sync_directory(path.parent)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


@contextmanager
def writer_lock(root: Path, run_id: str):
    try:
        import fcntl
    except ImportError as exc:
        raise CheckpointError("Checkpoint ownership requires a local filesystem with POSIX flock support") from exc
    durable_directory(root)
    path = root / f"{run_id}.lock"
    with path.open("a+b") as handle:
        os.chmod(path, 0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CheckpointError(f"Run '{run_id}' has an active writer; wait for it to exit") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
