"""Make the package importable for every test file, however it is run, and
keep GPU tests off a GPU another run already owns.

There is no editable install (main.py says so), so `preprocessing_annotations`
resolves only if src/ is on sys.path. Each test file used to add it itself,
and 14 of them did not -- they imported fine in a full run only because an
alphabetically earlier file had already done it. Run alone (`pytest
test/test_attachment_policy.py`), with `-k`, or under xdist, they died at
collection with ModuleNotFoundError. Measured 2026-11-04: all 14 passed alone
the moment src/ was on the path and nothing else changed.

One place, so the path stops depending on collection order. pytest loads this
file before collecting any sibling, so it also covers files added later.
"""

import contextlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def gpu_run_lock(request):
    """A `slow` test holds the pipeline's run lock, or skips if a run holds it.

    One T4 serves every session here. The pipeline, both eval probes and the
    trainer already serialize on run_lock.single_instance, but `slow` tests
    load their own models beside whatever holds it: on 2026-11-04 15:52Z a
    SAM3 integration test ran next to rebaseline_1275.sh and the chain's
    pipeline died reloading the 8B mid-run ("Some modules are dispatched on
    the CPU or the disk"), 29 minutes in, taking the whole ~40h chain with it.

    Skipping is the right outcome, not waiting: a chain holds the lock for
    hours, and a test suite that blocks on it looks hung. The lock is the
    existing one (FLOORPLAN_LOCK honoured), so a test and a run cannot
    disagree about what "busy" means. Every other test is untouched -- no
    lock, no skip.

    Only acquisition is guarded: single_instance(wait=False) signals a busy
    lock by exiting, and catching that around the test body too would turn a
    test's own SystemExit into a silent skip.
    """
    if "slow" not in request.keywords:
        yield
        return

    from preprocessing_annotations.orchestration.run_lock import single_instance

    with contextlib.ExitStack() as held:
        try:
            held.enter_context(single_instance(wait=False))
        except SystemExit:
            pytest.skip("GPU run lock held by another run; skipping GPU test")
        yield
