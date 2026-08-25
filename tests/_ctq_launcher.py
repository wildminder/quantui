"""Real-child launcher for the ctq MAIN-quant progress-bar regression.

Emits ctq-style per-tensor progress exactly as the convert_to_quant library does for
its MAIN quantization phase (verified against a real run debug log):

    (1/3) Processing (INT8): layer.a
    ----
        - Tensor shape: [1, 2]
        - Trying svd_lowrank ...
    (2/3) Processing (INT8): layer.b
    ----
        - Tensor shape: [3, 4]
        - Trying svd_lowrank ...
    (3/3) Processing (INT8): layer.c
    ----
        - Tensor shape: [5, 6]
        - Trying svd_lowrank ...

Crucially there is NO trailing "=== ... finished ===" clear line: real ctq ends on the
per-tensor details. That is precisely the condition that exposed the "no bar + fewer
strings" bug, so the regression tests below must observe the live bar surviving the
"----" separators.

Run with ``python -u`` (injected by QuantApp._run_proc) so the lines are not
block-buffered; the parent (worker_runner.WorkerRunner) reads them live off the raw pipe fd.
"""
import sys
import time

LAYERS = [
    ("layer.a", "[1, 2]"),
    ("layer.b", "[3, 4]"),
    ("layer.c", "[5, 6]"),
]


def main() -> None:
    total = len(LAYERS)
    for i, (name, shape) in enumerate(LAYERS, start=1):
        sys.stdout.write(f"({i}/{total}) Processing (INT8): {name}\n")
        sys.stdout.write("----\n")
        sys.stdout.write(f"    - Tensor shape: {shape}\n")
        sys.stdout.write("    - Trying svd_lowrank ...\n")
        sys.stdout.flush()
        time.sleep(0.03)
    # Intentionally NO trailing "=== Quantization finished successfully ===" line:
    # real convert_to_quant ends on the per-tensor details.


if __name__ == "__main__":
    main()
