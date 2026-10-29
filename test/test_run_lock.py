"""The run lock's contract: one holder at a time, and --no-wait's documented
exit code 3 when another run is active."""

import subprocess
import sys
import textwrap
from pathlib import Path

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
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(SRC)!r})
            from preprocessing_annotations.orchestration.run_lock import single_instance
            with single_instance(wait=True, lock_path={str(lock)!r}):
                print("held", flush=True)
                time.sleep(30)
        """)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"

        result = _child(lock, wait=False)

        assert result.returncode == 3, (result.returncode, result.stderr)
        assert "ValueError" not in result.stderr
    finally:
        holder.kill()
        holder.wait()
