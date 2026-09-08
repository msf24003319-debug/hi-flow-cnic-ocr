"""Exercise the REAL FastAPI /verify-cnic route (real PaddleOCR, no mocks)
via FastAPI TestClient, plus a couple of failure-mode checks.

Usage:  python tests/run_http_matrix.py
"""
from __future__ import annotations

import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SUPABASE_URL", "http://unused.local")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "unused")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)
FIX = os.path.join(os.path.dirname(__file__), "fixtures")
CNIC_A = "3520112345671"


def b64(name: str) -> str:
    with open(os.path.join(FIX, name), "rb") as fh:
        return base64.b64encode(fh.read()).decode()


CASES = [
    ("cnic_valid_A.png", CNIC_A, "correct + matching"),
    ("cnic_valid_B.png", CNIC_A, "different person's CNIC"),
    ("random_photo.png", CNIC_A, "random photo"),
    ("selfie.png", CNIC_A, "selfie"),
    ("passport.png", CNIC_A, "passport"),
    ("license.png", CNIC_A, "driving license"),
    ("cnic_blurry.png", CNIC_A, "blurry CNIC"),
    ("cnic_rotated.png", CNIC_A, "rotated CNIC"),
    ("cnic_illegible.png", CNIC_A, "illegible CNIC"),
]


def main() -> None:
    print("== POST /verify-cnic (real route, real PaddleOCR) ==")
    for fixture, entered, label in CASES:
        if not os.path.exists(os.path.join(FIX, fixture)):
            print(f"  {label:<28} FIXTURE MISSING")
            continue
        r = client.post(
            "/verify-cnic", json={"entered_cnic": entered, "image_base64": b64(fixture)}
        )
        body = r.json()
        allowed = r.status_code == 200 and body.get("match") is True
        print(
            f"  {label:<28} HTTP {r.status_code}  "
            f"status={body.get('status') or body.get('detail')!r}  "
            f"match={body.get('match')}  gate={'ALLOW' if allowed else 'BLOCK'}"
        )

    print("\n== failure modes ==")
    # corrupt / non-image bytes
    r = client.post(
        "/verify-cnic",
        json={"entered_cnic": CNIC_A, "image_base64": base64.b64encode(b"not an image at all" * 40).decode()},
    )
    print(f"  corrupt image bytes           HTTP {r.status_code}  {r.json().get('detail')!r}  gate=BLOCK")

    # bad base64
    r = client.post("/verify-cnic", json={"entered_cnic": CNIC_A, "image_base64": "%%%not-base64%%%" * 8})
    print(f"  invalid base64                HTTP {r.status_code}  {r.json().get('detail')!r}  gate=BLOCK")

    # entered CNIC not 13 digits
    r = client.post("/verify-cnic", json={"entered_cnic": "123", "image_base64": b64("cnic_valid_A.png")})
    print(f"  entered CNIC not 13 digits     HTTP {r.status_code}  {r.json().get('detail')!r}  gate=BLOCK")

    # commit endpoint rejects a foreign front_path / no auth
    r = client.post(
        "/verify-cnic/commit",
        json={"entered_cnic": CNIC_A, "front_path": "someone-else/cnic_front.jpg"},
    )
    print(f"  /commit no auth               HTTP {r.status_code}  {r.json().get('detail')!r}")

    # health
    r = client.get("/health")
    print(f"  /health                       HTTP {r.status_code}  {r.json()}")


if __name__ == "__main__":
    main()
