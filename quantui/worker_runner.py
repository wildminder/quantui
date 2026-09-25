"""Pure (Textual-free) worker subprocess runner (T05).

Launches a quantization worker as a separate Python process and streams its stdout
through :meth:`_pump`. The runner is **Textual-free**: it only calls the injected
``popen`` (defaults to ``subprocess.Popen``) and emits parsed :class:`StreamSegment`
objects to a :class:`~quantui.ui_bridge.LogObserver`. All widget writes happen
in the observer (``QuantApp``), which re-marshals to the main thread.

Threading / byte-path preserved EXACTLY from the pre-refactor ``app._run_proc`` +
``app._pump``:
* ``-u`` (unbuffered) is injected as ``argv[1]`` so tqdm / convert_to_quant frames
  stream live instead of being block-buffered until the bar finishes.
* Reads go through ``os.read(fd, 4096)`` on the **raw pipe fd** -- NOT
  ``proc.stdout.read(n)``, which would block until EOF and burst every frame at the end.
* ``stdout=PIPE, stderr=STDOUT`` so we see errors merged into the stream.
* When ``UNSLOTH_CTQ_DEBUG_PROGRESS=1``, raw bytes / verdicts are written to the
  provided ``progress_debug_fh``.
"""

import os
import subprocess
import threading
from collections.abc import Sequence

from .stream_parser import ProgressClassifier, StreamSegment, split_frames


def format_command_for_log(cmd: Sequence[str]) -> str:
    """Format a worker command for display without exposing its HF token."""
    display: list[str] = []
    redact_next = False
    for arg in cmd:
        if redact_next:
            display.append("********")
            redact_next = False
        elif arg == "--hf-token":
            display.append(arg)
            redact_next = True
        elif arg.startswith("--hf-token="):
            display.append("--hf-token=********")
        else:
            display.append(arg)
    return " ".join(display)


# Thread-pool env defaults for the worker subprocess. Torch/OMP default to the
# PHYSICAL core count (e.g. 12 on a 12c/24t Ryzen), leaving SMT threads idle. We
# only fill a var when the user has not set it explicitly (their value wins).
def _thread_env() -> dict:
    n = str(os.cpu_count() or 1)
    return {
        name: n
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        if name not in os.environ
    }


class WorkerRunner:
    """Runs a worker subprocess and pumps its stdout, emitting ``StreamSegment``s."""

    def __init__(self, popen=None) -> None:
        # Resolve at call time (not as a default-arg captured at import) so tests can
        # DI by monkeypatching ``worker_runner.subprocess.Popen`` before constructing
        # the app. ``popen=None`` falls back to the real ``subprocess.Popen``.
        self._popen = popen or subprocess.Popen
        # Handle to the live worker subprocess (None when idle). Guarded by ``_lock``
        # because the main thread (Stop button -> terminate) and the worker thread
        # (inside _pump) may touch it concurrently.
        self._proc = None  # type: ignore[var-annotated]
        self._lock = threading.Lock()

    def run(
        self,
        cmd: list[str],
        cwd: str,
        observer: "object",
        progress_debug_fh=None,
    ) -> int:
        """Launch ``cmd`` (inject ``-u`` as argv[1] if missing), pump, wait, return rc.

        ``observer`` must implement the ``LogObserver`` protocol (``on_segment``,
        ``on_raw``, ``live_count``); ``QuantApp`` is the concrete implementation.
        """
        if len(cmd) >= 2 and cmd[1] != "-u":
            cmd = [cmd[0], "-u", *cmd[1:]]
        # Echo the launched command as a VISIBLE line -- reproduces the old
        # `log_msg("$ <cmd>")`: it clears an (empty) live-progress bar (benign) and
        # is written to the run log + RichLog so the user sees what was launched.
        # Redact secrets for display only; Popen below receives the original argv.
        display_cmd = f"$ {format_command_for_log(cmd)}"
        observer.on_segment(
            StreamSegment(
                display_cmd,
                ProgressClassifier.classify(display_cmd),
                False,
            )
        )
        proc = self._popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env={**os.environ, **_thread_env()},
        )
        # Expose the live handle so the UI can terminate the worker on Stop. A short
        # critical section protects ``self._proc`` against concurrent access from the
        # main thread (terminate) while the worker thread is inside _pump.
        with self._lock:
            self._proc = proc
        try:
            rc = self._pump(proc, observer, progress_debug_fh)
        finally:
            # The child has exited by the time _pump returns (proc.wait() inside it),
            # so drop the handle -- unless a newer run already replaced it.
            with self._lock:
                if self._proc is proc:
                    self._proc = None
        return rc

    def _pump(self, proc, observer: "object", progress_debug_fh=None) -> int:
        """Read worker stdout in binary and emit each segment to ``observer.on_segment``.

        Splits the byte stream on both ``\\r`` (tqdm in-place rewrite -> live progress
        frame) and ``\\n`` (real log line), delivering each frame as it arrives. See the
        module docstring for why ``os.read`` (not ``proc.stdout.read``) is mandatory.
        """
        fd = proc.stdout.fileno()
        partial = ""
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break  # EOF: child closed stdout
            if progress_debug_fh is not None:
                try:
                    progress_debug_fh.write(f"RAW\t{chunk!r}\n")
                    progress_debug_fh.flush()
                except OSError:
                    pass  # debug trace sink closed mid-stream; streaming continues
            text = chunk.decode("utf-8", errors="replace")
            segments, partial = split_frames(text, partial)
            for content, _kind in segments:
                if not content.strip():
                    continue
                if progress_debug_fh is not None:
                    try:
                        progress_debug_fh.write(
                            f"SEG\t{ProgressClassifier.classify(content)}\t"
                            f"{observer.live_count()}\t{content!r}\n"
                        )
                        progress_debug_fh.flush()
                    except OSError:
                        pass  # debug trace sink closed mid-stream; streaming continues
                observer.on_segment(
                    StreamSegment(content, ProgressClassifier.classify(content), False)
                )
        if partial.strip():
            observer.on_segment(
                StreamSegment(partial, ProgressClassifier.classify(partial), False)
            )
        # Reap the child. If another thread (e.g. terminate()) already reaped it via
        # waitpid, wait() raises ChildProcessError; fall back to the cached returncode
        # so a user-initiated stop still returns a clean, non-zero rc instead of
        # crashing the worker thread (the stop UI transition still runs). The os.read
        # path above is untouched (T05 gate preserved).
        try:
            return proc.wait()
        except ChildProcessError:
            # Already reaped by terminate() on another thread; use the cached rc.
            return proc.returncode if proc.returncode is not None else -1

    def is_running(self) -> bool:
        """Return True iff a worker subprocess is currently alive.

        Reads ``self._proc`` under the lock; tolerates fakes that lack ``poll()``
        (treated as not-running).
        """
        with self._lock:
            proc = self._proc
        if proc is None:
            return False
        poll = getattr(proc, "poll", None)
        if poll is None:
            return False
        return poll() is None

    def terminate(self) -> None:
        """Terminate the running worker subprocess if any (best-effort, never raises).

        Sends SIGTERM (``proc.terminate()``; on Windows this is ``TerminateProcess``),
        waits briefly for a clean exit, then escalates to ``proc.kill()`` if it lingers.
        All subprocess interactions are wrapped so a stop can never bubble an exception
        into the UI / main thread.
        """
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        poll = getattr(proc, "poll", None)
        if poll is not None and poll() is not None:
            return  # already exited
        try:
            proc.terminate()
        except Exception:  # boundary: process may have died between poll() and
            return         # terminate(); nothing left to stop, never raise
        # Give it a moment to exit gracefully; if it does not, force-kill.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:  # boundary: kill raced with child exit; ignore
                pass
            try:
                proc.wait(timeout=5)
            except Exception:  # boundary: best-effort reap after kill; ignore
                pass
