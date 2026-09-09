"""Isolated PaddleOCR runtime probe — NO FastAPI, NO network.

Purpose: prove whether the PaddleOCR / PaddlePaddle native runtime can
initialise and run a single inference and exit cleanly, independently of
the web layer. If this segfaults, the crash is below the endpoint.

Run:  python tests/test_ocr_runtime.py
Never prints image bytes, base64, or any CNIC number.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "cnic_valid_A.png"


def main() -> int:
    print("=== PaddleOCR runtime probe ===")
    print(f"python           {sys.version.split()[0]}")
    try:
        import paddle  # noqa: WPS433

        print(f"paddlepaddle     {paddle.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"paddle import FAILED: {e!r}")
        return 1
    try:
        import paddleocr  # noqa: WPS433

        print(f"paddleocr        {paddleocr.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"paddleocr import FAILED: {e!r}")
        return 1
    try:
        import cv2  # noqa: WPS433

        print(f"opencv           {cv2.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"cv2 import FAILED: {e!r}")
        return 1
    import numpy as np

    print(f"numpy            {np.__version__}")
    print(
        "threads env      "
        f"OMP={os.getenv('OMP_NUM_THREADS')} MKL={os.getenv('MKL_NUM_THREADS')} "
        f"OPENBLAS={os.getenv('OPENBLAS_NUM_THREADS')} CPU_NUM={os.getenv('CPU_NUM')} "
        f"FLAGS_use_mkldnn={os.getenv('FLAGS_use_mkldnn')}"
    )

    # 1. paddle's own self-check (exercises a native op).
    print("\n[1] paddle.utils.run_check() ...", flush=True)
    t = time.time()
    try:
        paddle.utils.run_check()
    except Exception as e:  # noqa: BLE001
        print(f"    run_check raised: {e!r}")
    print(f"    done in {time.time() - t:.1f}s")

    # 2. Initialise the OCR pipeline exactly as the service does.
    print("\n[2] init OCR pipeline (through app.ocr._get_ocr) ...", flush=True)
    t = time.time()
    from app import ocr as ocr_mod

    engine = ocr_mod._get_ocr()  # noqa: SLF001 — intentional: same path as the service
    print(f"    init OK in {time.time() - t:.1f}s")

    # 3. One inference on a non-sensitive fixture.
    if not FIXTURE.exists():
        print(f"\n[3] SKIP — fixture missing: {FIXTURE.name}")
        return 0
    print(f"\n[3] single inference on {FIXTURE.name} ...", flush=True)
    data = FIXTURE.read_bytes()
    t = time.time()
    result = ocr_mod.run_ocr(data)
    dt = time.time() - t
    print(
        f"    inference OK in {dt:.1f}s — "
        f"candidates={len(result.candidates)} lines={result.line_count} "
        f"conf={result.confidence}"
    )

    # 4. Repeat inferences — the process must stay alive.
    print("\n[4] 3 more sequential inferences ...", flush=True)
    for i in range(3):
        t = time.time()
        ocr_mod.run_ocr(data)
        print(f"    run {i + 1}: {time.time() - t:.1f}s  (pid alive)")

    print("\n=== ALL STEPS COMPLETED, PROCESS EXITING NORMALLY ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
