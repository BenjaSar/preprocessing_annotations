"""The run lock's contract: one holder at a time, --no-wait's documented exit
code 3 when another run is active, and `slow` tests yielding the GPU to it."""

import contextlib
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SRC = PROJECT_ROOT / "src"


def _child(lock_path, wait, hold_seconds=0.0):
    """Run single_instance in a separate process, as real callers do."""
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(SRC)!r})
        from preprocessing_annotations.orchestration.run_lock import single_instance
        with single_instance(wait={wait}, lock_path={str(lock_path)!r}):
            time.sleep({hold_seconds})
        print("done")
    """)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_lock_is_acquired_when_free(tmp_path):
    result = _child(tmp_path / "lock", wait=True)
    assert result.returncode == 0, result.stderr
    assert "done" in result.stdout


def test_no_wait_exits_3_when_another_run_holds_the_lock(tmp_path):
    lock = tmp_path / "lock"
    with _holder(lock):
        result = _child(lock, wait=False)

        assert result.returncode == 3, (result.returncode, result.stderr)
        assert "ValueError" not in result.stderr


@contextlib.contextmanager
def _holder(lock_path):
    """A separate process holding `lock_path`, as a pipeline run would."""
    process = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(SRC)!r})
            from preprocessing_annotations.orchestration.run_lock import single_instance
            with single_instance(wait=True, lock_path={str(lock_path)!r}):
                print("held", flush=True)
                time.sleep(120)
        """)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "held"
        yield
    finally:
        process.kill()
        process.wait()


def _run_slow_test(lock_path):
    """Run one throwaway `slow` test under the real conftest, and report its
    outcome. It lives under test/ so test/conftest.py's fixture applies to it,
    and FLOORPLAN_LOCK points the fixture at this test's own lock."""
    directory = PROJECT_ROOT / "test" / "_gpu_lock_fixture_tmp"
    directory.mkdir()
    try:
        (directory / "test_needs_gpu.py").write_text(textwrap.dedent("""
            import pytest

            @pytest.mark.slow
            def test_pretends_to_use_the_gpu():
                pass
        """))
        return subprocess.run(
            # -rs prints each skip's reason, so a test can assert WHY it was
            # skipped rather than accepting any skip at all.
            [sys.executable, "-m", "pytest", str(directory), "-m", "slow",
             "-q", "-rs", "-p", "no:cacheprovider"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            env={**os.environ, "FLOORPLAN_LOCK": str(lock_path)},
        ).stdout
    finally:
        shutil.rmtree(directory)


def test_a_slow_test_skips_while_a_run_holds_the_lock(tmp_path):
    """The 2026-11-04 incident: a GPU test ran beside a chain's 8B pipeline
    run and killed it 29 minutes in. Yielding beats waiting -- a chain holds
    the lock for hours."""
    lock = tmp_path / "lock"
    with _holder(lock):
        output = _run_slow_test(lock)

    assert "1 skipped" in output, output
    assert "GPU run lock held" in output


def test_a_slow_test_runs_when_the_lock_is_free(tmp_path):
    """The guard must not skip the GPU suite whenever nothing holds the lock,
    which would turn every slow test into a silent no-op."""
    output = _run_slow_test(tmp_path / "lock")

    assert "1 passed" in output, output
