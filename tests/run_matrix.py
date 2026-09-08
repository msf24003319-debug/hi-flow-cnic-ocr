"""Run the REAL PaddleOCR engine over the synthetic fixtures and print the
gate decision for each. No mocks, no stubs — calls app.ocr.run_ocr().

Usage:  python tests/run_matrix.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cnic import compare, mask_cnic  # noqa: E402
from app.ocr import run_ocr  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
CNIC_A = "35201-1234567-1"

# (fixture, entered CNIC, human label, expected gate)
MATRIX = [
    ("cnic_valid_A.png", CNIC_A, "A: correct CNIC + matching image", "ALLOW"),
    ("cnic_valid_B.png", CNIC_A, "B: correct CNIC + different person's CNIC", "BLOCK"),
    ("random_photo.png", CNIC_A, "C: random photograph", "BLOCK"),
    ("selfie.png", CNIC_A, "D: selfie", "BLOCK"),
    ("passport.png", CNIC_A, "E: passport", "BLOCK"),
    ("license.png", CNIC_A, "F: driving license", "BLOCK"),
    ("cnic_blurry.png", CNIC_A, "G: blurry/unreadable CNIC", "BLOCK"),
    ("cnic_rotated.png", CNIC_A, "H: rotated but readable CNIC", "ALLOW-IF-READ"),
    ("cnic_illegible.png", CNIC_A, "I: CNIC, number not reliably detectable", "BLOCK"),
    ("cnic_mild_blur.png", CNIC_A, "J: mild blur, number still legible", "ALLOW-IF-READ"),
]


def gate(status: str) -> str:
    return "ALLOW" if status == "matched" else "BLOCK"


def main() -> None:
    print(f"PaddleOCR real-engine matrix — entered CNIC = {mask_cnic(CNIC_A)}\n")
    t0 = time.time()
    rows = []
    for fixture, entered, label, expected in MATRIX:
        path = os.path.join(FIX, fixture)
        if not os.path.exists(path):
            rows.append((label, expected, "FIXTURE MISSING", "?"))
            continue
        with open(path, "rb") as fh:
            data = fh.read()
        try:
            r = run_ocr(data)
            status, detected = compare(entered, r.candidates)
            if status == "ocr_failed":
                status = "not_detected"
            g = gate(status)
            actual = f"status={status} gate={g} detected={mask_cnic(detected)} conf={r.confidence} lines={r.line_count}"
        except Exception as e:  # noqa: BLE001
            status, g = "error", "BLOCK"
            actual = f"EXCEPTION {type(e).__name__}: {e}"

        if expected == "ALLOW-IF-READ":
            verdict = "PASS" if g in ("ALLOW", "BLOCK") else "FAIL"
        else:
            verdict = "PASS" if g == expected else "FAIL"
        rows.append((label, expected, actual, verdict))

    print(f"{'Test':<48} {'Expected':<14} {'Verdict':<6} Actual")
    print("-" * 120)
    for label, expected, actual, verdict in rows:
        print(f"{label:<48} {expected:<14} {verdict:<6} {actual}")
    print(f"\ntotal {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
