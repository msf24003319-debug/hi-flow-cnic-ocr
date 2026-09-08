"""CNIC number normalization + entered-vs-detected comparison.

Pure functions, no I/O — safe to unit test in isolation.
"""
from __future__ import annotations

import re

# 5 digits - 7 digits - 1 digit, in the image the separators may be missing,
# spaces, en-dashes, dots, etc. We also tolerate OCR splitting it across
# whitespace runs.
_CNIC_RE = re.compile(r"(\d{5})\s*[-–—.\s]?\s*(\d{7})\s*[-–—.\s]?\s*(\d)")

# Fallback: any unbroken run of exactly 13 digits.
_THIRTEEN_RE = re.compile(r"(?<!\d)(\d{13})(?!\d)")


def normalize_cnic(value: str | None) -> str:
    """Strip everything but digits. Returns '' for None / no digits."""
    if not value:
        return ""
    return re.sub(r"\D", "", value)


def is_valid_cnic(value: str | None) -> bool:
    return len(normalize_cnic(value)) == 13


def extract_cnic_candidates(text: str) -> list[str]:
    """Every distinct 13-digit CNIC-looking number found in `text`, in order."""
    seen: list[str] = []

    def _add(candidate: str) -> None:
        norm = normalize_cnic(candidate)
        if len(norm) == 13 and norm not in seen:
            seen.append(norm)

    for m in _CNIC_RE.finditer(text):
        _add("".join(m.groups()))
    for m in _THIRTEEN_RE.finditer(re.sub(r"[\s\-–—.]", "", text)):
        _add(m.group(1))

    return seen


def mask_cnic(value: str | None) -> str:
    """`*******-****-3` style — safe to log / return to the client."""
    norm = normalize_cnic(value)
    if len(norm) != 13:
        return "—"
    return f"*****-*****{norm[-3:-1]}-{norm[-1]}"


# OCR result -> profiles.cnic_ocr_status value.
STATUS_MATCHED = "matched"
STATUS_MISMATCH = "mismatch"
STATUS_OCR_FAILED = "ocr_failed"


def compare(entered: str, detected_candidates: list[str]) -> tuple[str, str | None]:
    """Return (status, detected_number).

    - matched    : the entered CNIC is among the numbers read off the image
    - mismatch   : a CNIC-looking number was read, but none equals the entered one
    - ocr_failed : no CNIC-looking number could be read at all
    """
    entered_norm = normalize_cnic(entered)
    if not detected_candidates:
        return STATUS_OCR_FAILED, None
    if entered_norm and entered_norm in detected_candidates:
        return STATUS_MATCHED, entered_norm
    return STATUS_MISMATCH, detected_candidates[0]
