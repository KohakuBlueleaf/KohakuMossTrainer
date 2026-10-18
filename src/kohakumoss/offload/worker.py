"""One background thread running queued host jobs in order. It stops on `close()` or
when its owner releases it. See docs/internals/offload.md (host workers)."""

import queue
import threading
import weakref
from collections.abc import Callable

JOIN_TIMEOUT = 5.0
_STOP = object()


def _serve(jobs: queue.Queue, errors: list) -> None:
    while True:
        job = jobs.get()
        try:
            if job is _STOP:
                return
            job()
        except BaseException as err:
            errors.append(err)
        finally:
            jobs.task_done()


def _stop(jobs: queue.Queue, thread: threading.Thread, timeout: float) -> None:
    """Post the sentinel and wait (bounded) for the thread to leave."""
    if not thread.is_alive():
        return
    jobs.put(_STOP)
    if thread is not threading.current_thread():
        thread.join(timeout)


class HostWorker:
    """`put(job)` queues a no-argument callable; `join()` waits for every queued job and
    raises the first error a job hit."""

    def __init__(self, name: str = "kohakumoss-host-worker"):
        self.jobs: queue.Queue = queue.Queue()
        self.errors: list[BaseException] = []
        self.thread = threading.Thread(
            target=_serve, args=(self.jobs, self.errors), daemon=True, name=name
        )
        self.thread.start()
        self._finalizer = weakref.finalize(
            self, _stop, self.jobs, self.thread, JOIN_TIMEOUT
        )

    def put(self, job: Callable[[], None]) -> None:
        self.jobs.put(job)

    def join(self) -> None:
        self.jobs.join()
        if self.errors:
            raise self.errors.pop(0)

    def close(self) -> None:
        """Stop the thread (idempotent; queued jobs run first)."""
        self._finalizer()
