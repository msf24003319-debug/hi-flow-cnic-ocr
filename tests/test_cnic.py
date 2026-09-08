"""Pure-logic tests — no PaddleOCR, no network. Run: python -m pytest"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cnic import (  # noqa: E402
    compare,
    extract_cnic_candidates,
    is_valid_cnic,
    mask_cnic,
    normalize_cnic,
)


def test_normalize():
    assert normalize_cnic("35202-1234567-1") == "3520212345671"
    assert normalize_cnic("3520212345671") == "3520212345671"
    assert normalize_cnic("  352 021 234 567 1 ") == "3520212345671"
    assert normalize_cnic(None) == ""


def test_is_valid():
    assert is_valid_cnic("35202-1234567-1")
    assert not is_valid_cnic("123")
    assert not is_valid_cnic("352021234567")  # 12 digits


def test_extract_dashed_and_plain():
    assert extract_cnic_candidates("Name\nCNIC 35202-1234567-1\nDOB") == ["3520212345671"]
    assert extract_cnic_candidates("id 3520212345671 issued") == ["3520212345671"]
    assert extract_cnic_candidates("35202 1234567 1") == ["3520212345671"]


def test_extract_none():
    assert extract_cnic_candidates("no numbers here") == []
    assert extract_cnic_candidates("phone 0300 1234567") == []


def test_compare_matched():
    status, detected = compare("35202-1234567-1", ["3520212345671"])
    assert status == "matched"
    assert detected == "3520212345671"


def test_compare_mismatch():
    status, detected = compare("35202-1234567-1", ["9999912345671"])
    assert status == "mismatch"
    assert detected == "9999912345671"


def test_compare_ocr_failed():
    status, detected = compare("35202-1234567-1", [])
    assert status == "ocr_failed"
    assert detected is None


def test_mask():
    assert mask_cnic("3520212345671") == "*****-*****67-1"
    assert mask_cnic("bad") == "—"
