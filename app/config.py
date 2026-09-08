"""Runtime configuration, read once from the environment.

Every secret here is BACKEND-ONLY. None of these values may ever be shipped
in the mobile app (no EXPO_PUBLIC_* equivalent) — the phone only ever learns
this service's public URL.
"""
from __future__ import annotations

import os


def _clean_url(v: str) -> str:
    return v.strip().rstrip("/")


SUPABASE_URL: str = _clean_url(os.getenv("SUPABASE_URL", ""))
SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

# The private bucket the mobile app uploads CNIC front/back images into.
CNIC_BUCKET: str = os.getenv("CNIC_BUCKET", "cnic-documents").strip()

# PaddleOCR language pack. Pakistani CNICs print the number in Latin digits,
# so "en" is correct and keeps the model small.
OCR_LANG: str = os.getenv("OCR_LANG", "en").strip()

# Reject oversized downloads early (the mobile client compresses to ~1280px
# JPEG, well under this).
MAX_IMAGE_MB: float = float(os.getenv("MAX_IMAGE_MB", "10"))

# Optional CORS allow-list (comma separated). The React Native client does
# not send an Origin header, so this only matters if a browser ever calls
# the service. Empty => no CORS headers.
ALLOWED_ORIGINS: list[str] = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()
]

# Hard ceiling on OCR wall-time so a pathological image can't pin a worker.
OCR_TIMEOUT_SECONDS: float = float(os.getenv("OCR_TIMEOUT_SECONDS", "25"))

# In-process rate limit for the unauthenticated /verify-cnic gate.
GATE_RATE_LIMIT: int = int(os.getenv("GATE_RATE_LIMIT", "20"))
GATE_RATE_PERIOD: float = float(os.getenv("GATE_RATE_PERIOD_SECONDS", "300"))


def assert_ready() -> None:
    """Everything needed for the authenticated /verify-cnic/commit endpoint."""
    missing = [
        name
        for name, val in (
            ("SUPABASE_URL", SUPABASE_URL),
            ("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )


def assert_ready_gate() -> None:
    """The /verify-cnic gate only runs OCR — it needs no Supabase creds."""
    return None
