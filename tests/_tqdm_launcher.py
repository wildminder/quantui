"""Real-child launcher used by the faithful live-progress reproduction.

Emits tqdm-style carriage-return-rewritten frames to stdout (exactly how a
third-party quant library writes its optimizer/progress bar), then a trailing
newline. Run with ``python -u`` so the frames are NOT block-buffered; the
parent (worker_runner.WorkerRunner) reads them live off the raw pipe fd.
"""
import sys
import time

PREFIX = "Optimizing INT8 (Prodigy-plateau): "


def frame(pct: int, n: int, total: int, sec: int) -> str:
    filled = pct // 10
    bar = "#" * filled + " " * (10 - filled)
    return f"{PREFIX}{pct}%|{bar}| {n}/{total} [00:0{sec}<?, ?it/s]"


def main() -> None:
    total = 10
    frames = [frame(i * 10, i, total, i) for i in range(total + 1)]
    for i, f in enumerate(frames):
        # First frame has no leading \r (already at column 0); later frames
        # return to column 0 with \r before re-writing the bar.
        sys.stdout.write(f if i == 0 else "\r" + f)
        sys.stdout.flush()
        time.sleep(0.03)
    sys.stdout.write("\n")
    sys.stdout.flush()
    # Follow the bar with a couple of ordinary log lines so we can observe the
    # clear-and-route behaviour too.
    sys.stdout.write("step a complete\n")
    sys.stdout.write("step b complete\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
