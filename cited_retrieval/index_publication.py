"""Index destination ownership, separate from source/corpus trust validation."""

import fcntl
import hashlib
import os
import stat
from pathlib import Path

Snapshot = tuple[int, int, int, int, int, str] | None


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def snapshot(target: Path) -> Snapshot:
    try:
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError:
        raise ValueError("index_changed") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("index_changed")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(fd)
        if _identity(before) != _identity(after) or _identity(target.lstat()) != _identity(after):
            raise ValueError("index_changed")
        return (*_identity(after), digest)
    except OSError:
        raise ValueError("index_changed") from None
    finally:
        os.close(fd)


def publish_index(temporary: Path, target: Path, expected: Snapshot) -> None:
    # Lock the existing directory inode, not a sidecar path that could collide with
    # source/crop/layout inputs or be unlinked and recreated by another builder.
    # All cooperative index publishers serialize their compare+publication here.
    fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if snapshot(target) != expected:
            raise ValueError("index_changed")
        if expected is None:
            try:
                # Atomic no-clobber also protects against noncooperative creators
                # racing the ownership check, including symlinks and directories.
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError:
                raise ValueError("index_changed") from None
        else:
            os.replace(temporary, target)
    finally:
        os.close(fd)
