"""Single-instance guard. Serialize pipeline runs to protect host RAM.

VLM + SAM each load multi-GB models. Two concurrent runs exhaust 15 GiB
RAM + swap and freeze the machine. This lock enforces one run at a time.
"""
import fcntl
import logging
import os
import sys
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DEFAULT_LOCK = os.environ.get("FLOORPLAN_LOCK", "/tmp/floorplan_pipeline.lock")


@contextmanager
def single_instance(wait: bool = True, poll: float = 5.0, lock_path: str = DEFAULT_LOCK):
    """Acquire an exclusive process-wide lock.

    Args:
        wait: If True, block until the lock is free. If False, exit(3) when busy.
        poll: Unused; kept for API compatibility. fcntl blocks natively.
        lock_path: Lock file path. Shared across all pipeline invocations.

    The lock auto-releases if this process dies (kernel closes the fd), so a
    crashed or OOM-killed run never leaves a stale lock behind.
    """
    fd = open(lock_path, "w")
    try:
        if wait:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.warning(
                    "Another pipeline run holds %s. Waiting for it to finish...", lock_path
                )
                start = time.time()
                fcntl.flock(fd, fcntl.LOCK_EX)  # blocks until other process releases
                logger.info("Lock acquired after %.0fs wait.", time.time() - start)
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.error(
                    "Another pipeline run is active (%s). Aborting (--no-wait).", lock_path
                )
                fd.close()
                sys.exit(3)
        fd.write(str(os.getpid()))
        fd.flush()
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
