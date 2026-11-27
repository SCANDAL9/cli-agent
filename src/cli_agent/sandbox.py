"""Bubblewrap command construction for untrusted local shell operations."""

from __future__ import annotations

import os
from pathlib import Path

SYSTEM_ROOTS = (
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/nix/store"),
    Path("/run/current-system"),
)
SAFE_PATHS = (
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
    "/run/current-system/sw/bin",
    "/nix/var/nix/profiles/default/bin",
)
MASKED_PATHS = (
    Path("/tmp"),
    Path("/home"),
    Path("/root"),
    Path("/run"),
    Path("/mnt"),
    Path("/media"),
)
MAX_CPU_SECONDS = 300
MAX_VIRTUAL_MEMORY_KIB = 4 * 1024 * 1024
MAX_FILE_SIZE_BLOCKS = 1024 * 1024


class SandboxUnavailableError(RuntimeError):
    """Raised when an OS sandbox is required but unavailable or misconfigured."""


def safe_command_path() -> str:
    """Returns only fixed, system-owned executable directories."""
    return os.pathsep.join(path for path in SAFE_PATHS if Path(path).is_dir())


def sandbox_environment(*, home: str = "/tmp/cli-agent-home") -> dict[str, str]:
    """Builds a minimal environment without inheriting the user's credentials."""
    env = {
        "PATH": safe_command_path(),
        "HOME": home,
        "TMPDIR": "/tmp",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in ("LANG", "LC_ALL", "TERM"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _add_dir(argv: list[str], created: set[Path], path: Path) -> None:
    """Creates missing destination directories in the otherwise empty namespace."""
    path = Path(os.path.normpath(path))
    for parent in reversed(path.parents):
        if parent == Path("/") or parent in created:
            continue
        argv.extend(("--dir", str(parent)))
        created.add(parent)
    if path != Path("/") and path not in created:
        argv.extend(("--dir", str(path)))
        created.add(path)


def _add_ro_bind(argv: list[str], created: set[Path], source: Path) -> None:
    """Bind one system-owned runtime directory read-only when it exists."""
    if not source.exists():
        return
    source = Path(os.path.normpath(source))
    _add_dir(argv, created, source.parent)
    argv.extend(("--ro-bind-try", str(source), str(source)))
    created.add(source)


def _git_metadata_paths(workspace: Path) -> tuple[Path, ...]:
    marker = workspace / ".git"
    try:
        paths: list[Path] = []
        if marker.is_dir():
            git_dir = marker.resolve()
        elif marker.is_file():
            paths.append(marker.resolve())
            line = marker.read_text(encoding="utf-8").strip()
            prefix = "gitdir: "
            if not line.lower().startswith(prefix):
                return tuple(paths)
            git_dir = Path(line[len(prefix) :])
            if not git_dir.is_absolute():
                git_dir = (workspace / git_dir).resolve()
            else:
                git_dir = git_dir.resolve()
        else:
            return ()

        common_dir_file = git_dir / "commondir"
        if common_dir_file.is_file():
            common_dir = Path(common_dir_file.read_text(encoding="utf-8").strip())
            if not common_dir.is_absolute():
                common_dir = git_dir / common_dir
            paths.append(common_dir.resolve())
        paths.append(git_dir)
        return tuple(dict.fromkeys(paths))
    except (OSError, UnicodeError, ValueError):
        return ()


def _add_workspace_bind(
    argv: list[str],
    created: set[Path],
    workspace: Path,
    *,
    cwd: Path,
    sanitized_git_config: Path,
) -> None:
    workspace = workspace.resolve()
    cwd = cwd.resolve()
    if not workspace.is_dir() or not _path_is_under(cwd, workspace):
        raise SandboxUnavailableError("The command working directory is outside its workspace.")

    home = Path.home().resolve()
    if workspace == home or any(_path_is_under(mask, workspace) for mask in MASKED_PATHS):
        raise SandboxUnavailableError("Refusing to expose a host home or shared temporary root.")

    # Mount only the selected workspace as writable; masked parent directories are recreated.
    _add_dir(argv, created, workspace.parent)
    _add_dir(argv, created, workspace)
    argv.extend(("--bind", str(workspace), str(workspace)))

    git_paths = _git_metadata_paths(workspace)
    # Bind metadata read-only, and hide local config files because they may contain
    # embedded credentials or credential-helper commands.
    for git_path in sorted(git_paths, key=lambda item: len(item.parts)):
        if not git_path.exists():
            continue
        _add_dir(argv, created, git_path.parent)
        argv.extend(("--ro-bind-try", str(git_path), str(git_path)))
        created.add(git_path)
    for git_path in git_paths:
        if not git_path.is_dir():
            continue
        for config_name in ("config", "config.worktree"):
            config_path = git_path / config_name
            if config_path.is_file():
                argv.extend(("--ro-bind", str(sanitized_git_config), str(config_path)))

    if not _path_is_under(cwd, workspace):
        raise SandboxUnavailableError("The command working directory is outside its workspace.")


def _shell_wrapper(shell_path: str, command: str) -> list[str]:
    script = (
        "set -euo pipefail; "
        f"ulimit -t {MAX_CPU_SECONDS}; "
        f"ulimit -v {MAX_VIRTUAL_MEMORY_KIB}; "
        f"ulimit -f {MAX_FILE_SIZE_BLOCKS}; "
        "ulimit -c 0; "
        'exec "$1" -lc "$2"'
    )
    return [shell_path, "-c", script, "cli-agent-sandbox", shell_path, command]


def build_bwrap_argv(
    command: str,
    *,
    bwrap_path: str,
    shell_path: str,
    workspace: Path,
    cwd: Path,
    network_access: bool,
    sanitized_git_config: Path,
) -> list[str]:
    """Builds a minimal-root Bubblewrap command with a single writable workspace."""
    if not Path(bwrap_path).is_file() or not Path(shell_path).is_file():
        raise SandboxUnavailableError("Bubblewrap or the system shell is unavailable.")
    if not sanitized_git_config.is_file():
        raise SandboxUnavailableError("A sanitized Git config is unavailable for sandbox setup.")

    argv = [
        bwrap_path,
        "--die-with-parent",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
    ]
    if not network_access:
        argv.append("--unshare-net")

    created: set[Path] = {Path("/")}
    for path in MASKED_PATHS:
        _add_dir(argv, created, path)
        if path == Path("/tmp"):
            argv.extend(("--size", "536870912", "--tmpfs", str(path)))
        else:
            argv.extend(("--tmpfs", str(path)))
        created.add(path)

    # Bubblewrap starts with an empty root. These immutable system roots provide only the
    # executable and shared-library runtime needed by commands, not host user data.
    for root in SYSTEM_ROOTS:
        _add_ro_bind(argv, created, root)
    for path in (
        Path("/etc/ssl/certs"),
        Path("/etc/ca-certificates"),
        Path("/etc/hosts"),
        Path("/etc/resolv.conf"),
        Path("/etc/nsswitch.conf"),
        Path("/etc/localtime"),
    ):
        _add_ro_bind(argv, created, path)

    _add_dir(argv, created, Path("/dev"))
    argv.extend(("--dev", "/dev"))
    created.add(Path("/dev"))
    _add_dir(argv, created, Path("/proc"))
    argv.extend(("--proc", "/proc"))
    created.add(Path("/proc"))

    _add_workspace_bind(
        argv,
        created,
        workspace,
        cwd=cwd,
        sanitized_git_config=sanitized_git_config,
    )
    _add_dir(argv, created, Path("/tmp/cli-agent-home"))

    argv.extend(
        (
            "--clearenv",
            "--setenv",
            "PATH",
            safe_command_path(),
            "--setenv",
            "HOME",
            "/tmp/cli-agent-home",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "GIT_TERMINAL_PROMPT",
            "0",
            "--setenv",
            "GIT_CONFIG_NOSYSTEM",
            "1",
            "--setenv",
            "GIT_CONFIG_GLOBAL",
            "/dev/null",
            "--setenv",
            "GIT_OPTIONAL_LOCKS",
            "0",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
        )
    )
    for key in ("LANG", "LC_ALL", "TERM"):
        value = os.environ.get(key)
        if value:
            argv.extend(("--setenv", key, value))

    argv.extend(("--chdir", str(cwd), "--", *_shell_wrapper(shell_path, command)))
    return argv


def build_bwrap_probe_argv(*, bwrap_path: str, shell_path: str) -> list[str]:
    """Builds a harmless user-namespace probe using only the system runtime."""
    argv = [
        bwrap_path,
        "--die-with-parent",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
    ]
    created: set[Path] = {Path("/")}
    for path in MASKED_PATHS:
        _add_dir(argv, created, path)
        argv.extend(("--tmpfs", str(path)))
        created.add(path)
    for root in SYSTEM_ROOTS:
        _add_ro_bind(argv, created, root)
    _add_dir(argv, created, Path("/dev"))
    argv.extend(("--dev", "/dev"))
    _add_dir(argv, created, Path("/proc"))
    argv.extend(("--proc", "/proc"))
    argv.extend(("--", shell_path, "-c", "exit 0"))
    return argv


def reduced_shell_argv(*, shell_path: str, command: str) -> list[str]:
    """Runs a command in a warned, non-isolated POSIX fallback after explicit opt-in."""
    if not Path(shell_path).is_file():
        raise SandboxUnavailableError("A POSIX shell is unavailable for reduced-isolation mode.")
    return _shell_wrapper(shell_path, command)
