"""Tests for the Textual-free WorkerRunner (T05).

The runner launches a worker subprocess and pumps its stdout through ``os.read`` on the
raw pipe fd, emitting ``StreamSegment``s to a ``LogObserver``. These tests use dependency
injection: a fake ``popen`` (and, for the pump, a patched ``os.read``) so no real torch /
convert_to_quant is required.

T05 gate (plan §4.4 / §8): ``worker_runner.py`` must NOT contain ``proc.stdout.read`` --
every read goes through ``os.read(fd, 4096)`` on the raw pipe fd. ``test_pump_uses_os_read_
not_buffered_read`` enforces this at runtime.
"""

import os
import subprocess
import sys
import threading
import time

from quantui import worker_runner
from quantui.live_progress import LiveProgressStore
from quantui.stream_parser import StreamSegment
from quantui.ui_bridge import LogRouter


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _CollectingObserver:
    """Minimal LogObserver for runner-level tests: records raw/segment callbacks."""

    def __init__(self) -> None:
        self.segments: list[StreamSegment] = []
        self.raw: list[bytes] = []
        self.count = 0

    def on_raw(self, raw: bytes) -> None:
        self.raw.append(raw)

    def on_segment(self, seg: StreamSegment) -> None:
        self.segments.append(seg)

    def live_count(self) -> int:
        return self.count


class _RecordingSink:
    """Minimal LogSink that records every widget write (no Textual widgets)."""

    def __init__(self) -> None:
        self.updates: list[str] = []
        self.log_lines: list[str] = []
        self.run_log: list[str] = []
        self.clears = 0

    def write_run_log(self, line: str) -> None:
        self.run_log.append(line)

    def write_log(self, line: str) -> None:
        self.log_lines.append(line)

    def update_live(self, text: str) -> None:
        self.updates.append(text)

    def update_progress(self, states) -> None:
        self.updates.append(states)

    def clear_live(self) -> None:
        self.clears += 1


class _RouterObserver:
    """Routes each emitted segment through a LogRouter (mimics QuantApp.on_segment)."""

    def __init__(self, router: LogRouter) -> None:
        self.router = router

    def on_raw(self, raw: bytes) -> None:
        pass

    def on_segment(self, seg: StreamSegment) -> None:
        self.router.dispatch(seg)

    def live_count(self) -> int:
        return 0


class _PipeProc:
    """Fake Popen result whose stdout is a real os.pipe-backed fd (so os.read works)."""

    def __init__(self, payload: bytes) -> None:
        r, w = os.pipe()
        with os.fdopen(w, "wb") as fw:
            fw.write(payload)
        self.stdout = os.fdopen(r, "rb", 0)  # unbuffered so os.read(fd) sees it
        self.returncode = 0

    def wait(self) -> int:
        return 0


# --------------------------------------------------------------------------- #
# T05: -u injection + stdout/stderr merge (DI, no real subprocess)
# --------------------------------------------------------------------------- #
def test_injects_unbuffered_flag():
    # Root-cause fix for the frozen/blinking bar: the child must be started with '-u'
    # (unbuffered) so its library-internal tqdm frames stream live instead of being
    # block-buffered until the bar's final newline. WorkerRunner.run must inject '-u'
    # as argv[1]. Verified via DI.
    captured: dict = {}

    def fake_Popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = kwargs
        return _PipeProc(b"$ echo done\n=== finished ===\n")

    observer = _CollectingObserver()
    runner = worker_runner.WorkerRunner(popen=fake_Popen)
    rc = runner.run(["python", "worker.py", "--model", "x"], ".", observer=observer)

    assert rc == 0
    assert captured["cmd"][0] == "python"
    assert captured["cmd"][1] == "-u", captured["cmd"]
    assert captured["cmd"][2:] == ["worker.py", "--model", "x"], captured["cmd"]
    # stdout/stderr are merged and read in binary (we split on \r and \n ourselves).
    assert captured["kwargs"].get("stderr") == subprocess.STDOUT
    assert captured["kwargs"].get("stdout") == subprocess.PIPE
    # The launched command is echoed as a visible line segment.
    assert any(isinstance(s, StreamSegment) and s.content.startswith("$ python -u")
               for s in observer.segments)


def test_format_command_for_log_leaves_ordinary_command_unchanged():
    cmd = ["python", "-m", "quantui.worker", "--method", "q4_k_m"]

    assert worker_runner.format_command_for_log(cmd) == " ".join(cmd)


def test_format_command_for_log_masks_hf_token_forms_only():
    cmd = [
        "python",
        "--tokenizer-mode",
        "token",
        "--hf-token",
        "hf_secret_one",
        "--token-type",
        "write",
        "--hf-token=hf_secret_two",
        "--other-token=keep-me",
    ]

    formatted = worker_runner.format_command_for_log(cmd)

    assert "hf_secret_one" not in formatted
    assert "hf_secret_two" not in formatted
    assert "--hf-token ********" in formatted
    assert "--hf-token=********" in formatted
    assert "--tokenizer-mode token" in formatted
    assert "--token-type write" in formatted
    assert "--other-token=keep-me" in formatted


def test_run_redacts_display_but_popen_receives_original_hf_token():
    token = "hf_original_secret"
    captured: dict = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = kwargs
        return _PipeProc(b"=== finished ===\n")

    observer = _CollectingObserver()
    runner = worker_runner.WorkerRunner(popen=fake_popen)
    rc = runner.run(
        ["python", "worker.py", "--hf-token", token, "--other-token=visible"],
        ".",
        observer=observer,
    )

    assert rc == 0
    assert captured["cmd"] == [
        "python",
        "-u",
        "worker.py",
        "--hf-token",
        token,
        "--other-token=visible",
    ]
    logged = "\n".join(segment.content for segment in observer.segments)
    assert token not in logged
    assert "$ python -u worker.py --hf-token ******** --other-token=visible" in logged


def test_run_passes_thread_env(monkeypatch):
    # CPU-utilization fix: the worker subprocess must receive OMP/MKL/OPENBLAS
    # thread-pool env vars (full logical core count) unless the user set them.
    captured: dict = {}
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")  # user-set value must win

    def fake_Popen(cmd, **kwargs):
        captured["kwargs"] = kwargs
        return _PipeProc(b"=== finished ===\n")

    observer = _CollectingObserver()
    runner = worker_runner.WorkerRunner(popen=fake_Popen)
    rc = runner.run(["python", "worker.py"], ".", observer=observer)

    assert rc == 0
    env = captured["kwargs"]["env"]
    n = str(os.cpu_count())
    assert env["OMP_NUM_THREADS"] == n, env
    assert env["MKL_NUM_THREADS"] == n, env
    assert env["OPENBLAS_NUM_THREADS"] == "3", env  # user override preserved
    assert env["PATH"] == os.environ.get("PATH")  # rest of parent env inherited


# --------------------------------------------------------------------------- #
# T05: os.read pump delivers tqdm frames live (NOT proc.stdout.read)
# --------------------------------------------------------------------------- #
def test_pump_delivers_progress_frames_live(monkeypatch):
    # Regression for the "frozen 0% bar / empty placeholder" bug: the reader must stream
    # each '\r'-delimited tqdm frame as it arrives (via os.read on the raw pipe fd, which
    # returns available bytes immediately) and keep the LAST frame (100%), not freeze at
    # 0% or wait for EOF and burst through at the end.
    store = LiveProgressStore()
    sink = _RecordingSink()
    router = LogRouter(sink, store)
    observer = _RouterObserver(router)

    # Replay the worker's tqdm frames exactly as a pipe would deliver them: each
    # '\r'-separated frame arrives as its own os.read() return; the bar finishes with a
    # trailing '\n'. os.read is patched because _pump reads the raw fd directly.
    chunks = [
        b"Optimizing INT8 (Prodigy-plateau):   0%|          | 0/4000 [00:00<?, ?it/s]\r",
        b"Optimizing INT8 (Prodigy-plateau):  50%|#####     | 2000/4000 [00:01<?, ?it/s]\r",
        b"Optimizing INT8 (Prodigy-plateau): 100%|##########| 4000/4000 [00:02<?, ?it/s]\n",
    ]
    state = {"i": 0}

    def fake_read(fd, n):
        if state["i"] >= len(chunks):
            return b""
        c = chunks[state["i"]]
        state["i"] += 1
        return c

    monkeypatch.setattr(os, "read", fake_read)

    class _Stream:
        @staticmethod
        def fileno():
            return 0

    class _Proc:
        stdout = _Stream()
        returncode = 0

        def wait(self):
            return 0

    runner = worker_runner.WorkerRunner(popen=subprocess.Popen)  # unused (we call _pump)
    runner._pump(_Proc(), observer)

    # Every '\r' frame was delivered as it arrived; the stored value is the LAST frame
    # (100% / 4000 done), not frozen at the initial 0/4000 frame.
    assert store, "no progress captured"
    latest = next(iter(store.snapshot().values()))
    assert "4000/4000" in latest, latest
    assert " 0/4000" not in latest, latest

    # A subsequent real log line clears the progress and routes to the log sink.
    router.dispatch(StreamSegment("=== Quantization finished successfully ===", "line", False))
    assert store.snapshot() == {}
    assert len(sink.log_lines) >= 1


# --------------------------------------------------------------------------- #
# T05 gate: the pump must use os.read, never proc.stdout.read
# --------------------------------------------------------------------------- #
def test_pump_uses_os_read_not_buffered_read(monkeypatch):
    # Hard gate (plan §4.4): the pump must read via os.read on the raw pipe fd, never
    # proc.stdout.read (which would block until EOF). We back stdout with a _Stream that
    # has NO .read method -- so if _pump used proc.stdout.read it would raise and this
    # test would error out. Reaching the assertion below proves os.read was used.
    calls = {"os_read": 0}

    def fake_read(fd, n):
        calls["os_read"] += 1
        return b""  # immediate EOF -> loop ends

    monkeypatch.setattr(os, "read", fake_read)

    class _Stream:
        @staticmethod
        def fileno():
            return 0
        # NOTE: deliberately no .read method.

    class _Proc:
        stdout = _Stream()
        returncode = 0

        def wait(self):
            return 0

    observer = _CollectingObserver()
    runner = worker_runner.WorkerRunner(popen=subprocess.Popen)
    runner._pump(_Proc(), observer)
    assert calls["os_read"] >= 1, "worker pump did not call os.read"


# --------------------------------------------------------------------------- #
# Stop / terminate: terminate() kills a live worker and run() returns non-zero
# --------------------------------------------------------------------------- #
def test_terminate_stops_running_process():
    # Direct unit test of the public stop surface: a real sleeping child must be
    # terminated quickly, is_running() flips to False, and the exit code is non-zero.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    runner = worker_runner.WorkerRunner(popen=subprocess.Popen)
    runner._proc = proc  # simulate a run in progress (Stop button path)

    assert runner.is_running() is True
    runner.terminate()

    # The child must exit promptly (SIGTERM), not linger for the full 30s.
    rc = proc.wait(timeout=5)
    assert rc != 0, f"child should have been killed, got rc={rc}"
    assert runner.is_running() is False


def test_run_returns_nonzero_when_terminated():
    # End-to-end: drive run() on a real sleeping child in a background thread and
    # terminate it from the main thread. run() must return a non-zero rc and the
    # runner must report not-running afterwards. The os.read pump path is exercised
    # exactly as in production (T05 preserved: no proc.stdout.read).
    result: dict = {}
    runner = worker_runner.WorkerRunner(popen=subprocess.Popen)

    def run_in_thread() -> None:
        # A child that sleeps and writes nothing: the pump blocks on os.read until the
        # child is killed, then sees EOF and reaps it. The launched command is echoed
        # as a visible line (harmless for this test).
        result["rc"] = runner.run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            ".",
            observer=_CollectingObserver(),
        )

    t = threading.Thread(target=run_in_thread, daemon=True)
    t.start()

    # Wait until the child is actually launched and reported running.
    deadline = time.time() + 5
    while not runner.is_running() and time.time() < deadline:
        time.sleep(0.02)

    assert runner.is_running() is True, "child never reported running"
    runner.terminate()
    t.join(timeout=10)
    assert not t.is_alive(), "run() did not return after terminate()"
    assert result.get("rc") != 0, f"run() should return non-zero after stop, got {result.get('rc')}"
    assert runner.is_running() is False
