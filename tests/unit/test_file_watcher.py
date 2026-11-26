import sys
import time
from pathlib import Path

from cli_agent.code_graph import CodeGraphIndexer, CodeGraphStore
from cli_agent.file_watcher import FileWatcher


def test_linux_inotify_events_refresh_graph_without_poll_interval_delay(
    tmp_path: Path, monkeypatch
) -> None:
    if not sys.platform.startswith("linux"):
        return
    root = tmp_path / "project"
    root.mkdir()
    db = CodeGraphStore(tmp_path / "graph.db")
    watcher = FileWatcher(root, CodeGraphIndexer(db), "watch-test", interval=5.0)
    assert watcher.start()
    try:

        def unexpected_full_scan():
            raise AssertionError("normal native events must not scan the whole tree")

        monkeypatch.setattr(watcher, "_take_snapshot", unexpected_full_scan)
        source = root / "event.py"
        started = time.perf_counter()
        source.write_text("def created(): pass\n", encoding="utf-8")
        changes = watcher.wait(timeout=2.0)
        elapsed = time.perf_counter() - started
        assert changes and changes[0].kind == "created"
        assert db.get_file_symbols("watch-test", str(source))[0]["symbol_name"] == "created"
        assert elapsed < 1.0

        source.write_text("def modified(): pass\n", encoding="utf-8")
        assert [event.kind for event in watcher.wait(timeout=2.0)] == ["modified"]
        assert db.get_file_symbols("watch-test", str(source))[0]["symbol_name"] == "modified"

        source.unlink()
        assert [event.kind for event in watcher.wait(timeout=2.0)] == ["deleted"]
        assert db.get_file_symbols("watch-test", str(source)) == []
    finally:
        watcher.close()


def test_file_watcher_refreshes_changes_and_recovers_deletions(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    db = CodeGraphStore(tmp_path / "graph.db")
    indexer = CodeGraphIndexer(db)
    source = root / "one.py"
    source.write_text("def first():\n    pass\n", encoding="utf-8")
    watcher = FileWatcher(root, indexer, "watch-test")

    indexer.index_file(source, "watch-test")
    assert watcher.poll() == []
    assert db.get_stats("watch-test")["files"] == 1
    source.unlink()
    changes = watcher.recover()
    assert [change.kind for change in changes] == ["deleted"]
    assert db.get_stats("watch-test")["files"] == 0

    source.write_text("def first():\n    pass\n", encoding="utf-8")
    assert [event.kind for event in watcher.poll()] == ["created"]
    source.write_text("def second():\n    pass\n", encoding="utf-8")
    assert [event.kind for event in watcher.poll()] == ["modified"]
    assert [s["symbol_name"] for s in db.get_file_symbols("watch-test", str(source))] == ["second"]

    source.unlink()
    assert [event.kind for event in watcher.recover()] == ["deleted"]
    assert db.get_stats("watch-test") == {"files": 0, "symbols": 0, "edges": 0}


def test_file_watcher_renames_and_ignores_excluded_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    db = CodeGraphStore(tmp_path / "graph.db")
    indexer = CodeGraphIndexer(db)
    source = root / "before.py"
    source.write_text("def kept():\n    pass\n", encoding="utf-8")
    watcher = FileWatcher(root, indexer, "watch-test")
    watcher.poll()
    watcher.poll()
    source.rename(root / "after.py")

    events = watcher.poll()
    assert [(event.kind, event.previous_path, event.path) for event in events] == [
        ("renamed", source, root / "after.py")
    ]
    assert db.get_file_symbols("watch-test", str(source)) == []
    assert len(db.get_file_symbols("watch-test", str(root / "after.py"))) == 1

    ignored = root / ".git"
    ignored.mkdir()
    ignored_file = ignored / "ignored.py"
    ignored_file.write_text("def ignored(): pass\n", encoding="utf-8")
    assert watcher.poll() == []
    assert db.get_file_symbols("watch-test", str(ignored_file)) == []


def test_file_watcher_bounds_event_batches_without_losing_pending_events(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    watcher = FileWatcher(
        root, CodeGraphIndexer(CodeGraphStore(tmp_path / "graph.db")), "watch-test"
    )
    watcher.poll()
    for name in ("b.py", "a.py", "c.py"):
        (root / name).write_text("def f(): pass\n", encoding="utf-8")

    first = watcher.poll(max_events=1)
    remaining = watcher.poll(max_events=1) + watcher.poll(max_events=1)
    assert len(first) == 1
    assert len(remaining) == 2
    assert first[0].path.name == "a.py"
