"""Process shutdown of a run: SIGTERM ends the run like an exception (finally blocks and
exit handlers run), and at exit the run's multiprocessing children (loader workers) and
its forkserver end with it. See docs/concepts/training.md#shutdown."""

import atexit
import multiprocessing
import multiprocessing.forkserver
import os
import signal
import threading

_installed = False


def _forkserver_pid() -> int | None:
    """The pid of this process's multiprocessing forkserver (a private CPython field)."""
    return getattr(multiprocessing.forkserver._forkserver, "_forkserver_pid", None)


def stop_children() -> None:
    """Kill every live multiprocessing child of this process and its forkserver."""
    for child in multiprocessing.active_children():
        child.kill()
    pid = _forkserver_pid()
    if pid:
        try:
            os.kill(
                pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM
            )
        except OSError:
            pass


def _terminate(signum: int, frame) -> None:
    raise SystemExit(128 + signum)


def install() -> None:
    """Once per process, from the main thread: SIGTERM raises SystemExit(143) there,
    and `stop_children` runs at exit (before multiprocessing's own exit handler)."""
    global _installed
    if _installed or threading.current_thread() is not threading.main_thread():
        return
    _installed = True
    atexit.register(stop_children)
    signal.signal(signal.SIGTERM, _terminate)
