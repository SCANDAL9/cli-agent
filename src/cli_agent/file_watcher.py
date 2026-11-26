"""Deterministic stdlib polling watcher for incremental code-graph refreshes."""

from __future__ import annotations

import ctypes
import os
import select
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from .code_graph import EXCLUDED_DIRS, SUPPORTED_EXTENSIONS, CodeGraphIndexer


@dataclass(frozen=True)
class FileChange:
    """A source-file change; renames carry the prior path in ``previous_path``."""

    kind: str
    path: Path
    previous_path: Path | None = None


@dataclass(frozen=True)
class _FileState:
    mtime_ns: int
    size: int
    inode: int


class FileWatcher:
    """Poll a project tree and incrementally apply bounded changes to its graph.

    The first poll establishes a baseline without re-indexing the tree. ``recover``
    deliberately performs a complete comparison after missed events or process downtime.
    """

    def __init__(
        self,
        root: str | Path,
        indexer: CodeGraphIndexer,
        project_key: str,
        *,
        interval: float = 0.25,
    ) -> None:
        self.root = Path(root).resolve()
        self.indexer = indexer
        self.project_key = project_key
        self.interval = max(0.01, interval)
        self._snapshot: dict[Path, _FileState] | None = None
        self._pending: list[FileChange] = []
        self._native_fd: int | None = None
        self._native_watches: dict[int, Path] = {}
        self._native_pending: dict[Path, FileChange] = {}

    def _included(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return False
        return (
            path.suffix in SUPPORTED_EXTENSIONS
            and not any(
                part in EXCLUDED_DIRS or part.startswith(".") for part in relative.parts[:-1]
            )
            and not any(part.startswith(".") for part in relative.parts)
        )

    def _take_snapshot(self) -> dict[Path, _FileState]:
        result: dict[Path, _FileState] = {}
        if not self.root.is_dir():
            return result
        for current, dirs, files in os.walk(self.root):
            dirs[:] = sorted(
                name for name in dirs if name not in EXCLUDED_DIRS and not name.startswith(".")
            )
            for name in sorted(files):
                path = (Path(current) / name).resolve()
                if not self._included(path):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                result[path] = _FileState(stat.st_mtime_ns, stat.st_size, stat.st_ino)
        return result

    def poll(self, *, max_events: int = 100) -> list[FileChange]:
        """Detect and apply up to ``max_events`` changes in stable path order."""
        if max_events <= 0:
            return []
        current = self._take_snapshot()
        if self._snapshot is None:
            self._snapshot = current
            return []
        old = self._snapshot
        removed = set(old) - set(current)
        added = set(current) - set(old)
        changes: list[FileChange] = []

        # Pair same-inode moves deterministically; unmatched files become deletes/creates.
        old_by_inode: dict[int, list[Path]] = {}
        new_by_inode: dict[int, list[Path]] = {}
        for path in removed:
            old_by_inode.setdefault(old[path].inode, []).append(path)
        for path in added:
            new_by_inode.setdefault(current[path].inode, []).append(path)
        for inode in sorted(old_by_inode.keys() & new_by_inode.keys()):
            for previous, path in zip(
                sorted(old_by_inode[inode]), sorted(new_by_inode[inode]), strict=False
            ):
                removed.remove(previous)
                added.remove(path)
                changes.append(FileChange("renamed", path, previous))

        changes.extend(FileChange("deleted", path) for path in removed)
        changes.extend(FileChange("created", path) for path in added)
        changes.extend(
            FileChange("modified", path)
            for path in set(old) & set(current)
            if old[path] != current[path]
        )
        self._pending.extend(sorted(changes, key=lambda event: (str(event.path), event.kind)))
        self._snapshot = current
        batch, self._pending = self._pending[:max_events], self._pending[max_events:]
        for event in batch:
            if event.previous_path is not None:
                self.indexer.store.remove_file_symbols(self.project_key, str(event.previous_path))
            if event.kind == "deleted":
                self.indexer.store.remove_file_symbols(self.project_key, str(event.path))
            elif self._included(event.path) and event.path.is_file():
                self.indexer.index_file(event.path, self.project_key)
        return batch

    def start(self) -> bool:
        """Start native Linux inotify observation; return false when unavailable."""
        if self._native_fd is not None:
            return True
        if not sys.platform.startswith("linux"):
            self._snapshot = self._take_snapshot()
            return False
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        fd = init(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        if fd < 0:
            self._snapshot = self._take_snapshot()
            return False
        self._native_fd = fd
        self._snapshot = self._take_snapshot()
        self._sync_native_watches()
        return True

    def _sync_native_watches(self) -> None:
        """Watch every in-scope directory, dropping watches for removed directories."""
        if self._native_fd is None:
            return
        existing = set(self._native_watches.values())
        directories = {self.root}
        if self.root.is_dir():
            for current, dirs, _ in os.walk(self.root):
                dirs[:] = sorted(
                    name for name in dirs if name not in EXCLUDED_DIRS and not name.startswith(".")
                )
                directories.update((Path(current) / name).resolve() for name in dirs)
        libc = ctypes.CDLL(None, use_errno=True)
        add_watch = libc.inotify_add_watch
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        mask = 0x00000100 | 0x00000200 | 0x00000080 | 0x00000040 | 0x00000008 | 0x00000004
        for directory in sorted(directories - existing, key=str):
            wd = add_watch(self._native_fd, os.fsencode(directory), mask)
            if wd >= 0:
                self._native_watches[wd] = directory
        for wd, directory in list(self._native_watches.items()):
            if directory not in directories:
                self._native_watches.pop(wd, None)

    def wait(self, timeout: float | None = None, *, max_events: int = 100) -> list[FileChange]:
        """Wait for native activity and refresh only paths named by inotify.

        Linux inotify notifications wake the graph refresh immediately. Other platforms,
        or Linux systems without inotify support, retain the stdlib polling fallback.
        Inotify overflow is recovered by a full snapshot reconciliation.
        """
        if self._snapshot is None:
            self.start()
        wait_seconds = self.interval if timeout is None else max(0.0, timeout)
        if self._native_fd is None:
            if wait_seconds:
                select.select([], [], [], wait_seconds)
            self._sync_native_watches()
            return self.poll(max_events=max_events)
        overflow = False
        sync_watches = False
        if not self._native_pending:
            readable, _, _ = select.select([self._native_fd], [], [], wait_seconds)
        else:
            readable = [self._native_fd]
        if readable:
            try:
                data = os.read(self._native_fd, 65536)
            except BlockingIOError:
                data = b""
            overflow, sync_watches = self._queue_native_events(data)
        if overflow:
            changes = self.recover(max_events=max_events)
            self._sync_native_watches()
            return changes
        if sync_watches:
            self._sync_native_watches()
        return self._apply_native_pending(max_events)

    def _queue_native_events(self, data: bytes) -> tuple[bool, bool]:
        """Parse one bounded inotify read and coalesce it by affected source path."""
        moved_from: dict[int, Path] = {}
        moved_to: dict[int, Path] = {}
        other_events: list[tuple[int, int, int, Path]] = []
        sync_watches = False
        offset = 0
        while offset + 16 <= len(data):
            wd, mask, cookie, name_length = struct.unpack_from("iIII", data, offset)
            offset += 16
            raw_name = data[offset : offset + name_length].split(b"\0", 1)[0]
            offset += name_length
            if mask & 0x00004000:  # IN_Q_OVERFLOW
                return True, True
            directory = self._native_watches.get(wd)
            if directory is None:
                continue
            path = (directory / os.fsdecode(raw_name)).resolve() if raw_name else directory
            if mask & 0x40000000:  # IN_ISDIR
                sync_watches = True
                continue
            if not self._included(path):
                continue
            if mask & 0x00000040:  # IN_MOVED_FROM
                moved_from[cookie] = path
            elif mask & 0x00000080:  # IN_MOVED_TO
                moved_to[cookie] = path
            else:
                other_events.append((wd, mask, cookie, path))

        changes: dict[Path, FileChange] = {}
        for cookie in sorted(moved_from.keys() | moved_to.keys()):
            previous = moved_from.get(cookie)
            path = moved_to.get(cookie)
            if previous is not None and path is not None:
                changes[path] = FileChange("renamed", path, previous)
            elif previous is not None:
                changes[previous] = FileChange("deleted", previous)
            elif path is not None:
                changes[path] = FileChange("created", path)

        for _, mask, _, path in other_events:
            if mask & (0x00000200 | 0x00000400):  # IN_DELETE | IN_DELETE_SELF
                existing = changes.get(path) or self._native_pending.get(path)
                if existing is not None and existing.kind == "created":
                    changes.pop(path, None)
                else:
                    changes[path] = FileChange("deleted", path)
            elif mask & (0x00000100 | 0x00000008 | 0x00000004):  # CREATE | CLOSE_WRITE | ATTRIB
                existing = changes.get(path) or self._native_pending.get(path)
                if existing is not None and existing.kind in {"created", "renamed"}:
                    continue
                kind = "modified" if path in self._snapshot else "created"
                changes[path] = FileChange(kind, path)

        for path, event in changes.items():
            previous = self._native_pending.get(path)
            if previous is not None:
                if previous.kind == "created" and event.kind == "deleted":
                    self._native_pending.pop(path, None)
                    continue
                if previous.kind == "renamed" and event.kind == "modified":
                    continue
                if previous.kind == "created" and event.kind == "modified":
                    continue
            self._native_pending[path] = event
        return False, sync_watches

    def _apply_native_pending(self, max_events: int) -> list[FileChange]:
        if max_events <= 0:
            return []
        paths = sorted(self._native_pending, key=str)[:max_events]
        batch = [self._native_pending.pop(path) for path in paths]
        for event in batch:
            if event.previous_path is not None:
                self.indexer.store.remove_file_symbols(self.project_key, str(event.previous_path))
                if self._snapshot is not None:
                    self._snapshot.pop(event.previous_path, None)
            if event.kind == "deleted":
                self.indexer.store.remove_file_symbols(self.project_key, str(event.path))
                if self._snapshot is not None:
                    self._snapshot.pop(event.path, None)
            elif self._included(event.path) and event.path.is_file():
                self.indexer.index_file(event.path, self.project_key)
                try:
                    stat = event.path.stat()
                except OSError:
                    continue
                if self._snapshot is not None:
                    self._snapshot[event.path] = _FileState(
                        stat.st_mtime_ns, stat.st_size, stat.st_ino
                    )
        return batch

    def close(self) -> None:
        """Release the native watcher descriptor, if started."""
        if self._native_fd is not None:
            os.close(self._native_fd)
            self._native_fd = None
            self._native_watches.clear()

    def recover(self, *, max_events: int = 100) -> list[FileChange]:
        """Rescan after missed notifications; aliases poll for polling-based recovery."""
        return self.poll(max_events=max_events)

    def run(self, stop_event, *, max_events: int = 100) -> None:
        """Poll until a threading.Event-like object is set."""
        self.start()
        try:
            while not stop_event.is_set():
                self.wait(timeout=self.interval, max_events=max_events)
        finally:
            self.close()
