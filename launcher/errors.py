"""One exception type for every failure the user can act on."""

from __future__ import annotations


class LauncherError(Exception):
    """A failure with a message that names the thing, the cause and the next command."""

    code = 1


class UsageError(LauncherError):
    """The command line itself is wrong."""

    code = 2


def describe_os_error(e: OSError) -> str:
    """One line for an OSError that reached the top: the path when there is one, and the system's reason."""
    where = f"{e.filename}: " if e.filename else ""
    return f"{where}{e.strerror or e}. Check the path, its permissions and the free disk space, then run the command again."
