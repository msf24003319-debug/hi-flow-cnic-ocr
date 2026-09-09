"""CNIC OCR verification microservice.

Two endpoints:

1. POST /verify-cnic  (UNAUTHENTICATED, stateless — the REGISTRATION GATE)
     in : { entered_cnic, image_base64 }
     out: { status: matched|mismatch|not_detected, match, detected_cnic_masked, confidence }
   Runs PaddleOCR on the posted image, extracts a 13-digit CNIC, normalizes
   both numbers, compares. Writes nothing, stores nothing. The mobile app
   calls this BEFORE it creates the auth user and refuses to continue
   registration unless `status == "matched"`.

2. POST /verify-cnic/commit  (AUTHENTICATED — persistence for the admin queue)
     in : { entered_cnic, front_path }  + Authorization: Bearer <user JWT>
     out: { ok, status }
   Downloads the already-uploaded CNIC image from the private bucket with
   the service role, re-runs OCR, and writes profiles.cnic_ocr_* so the
   admin sees the result. Best-effort: the real gate already happened in (1).

This service NEVER approves or rejects an account and NEVER claims the CNIC
is genuine / government-issued. It only does text detection + number
comparison. The admin remains responsible for final manual verification
(profiles.verification_status), which this service does not touch.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import config
from .cnic import compare, is_valid_cnic, mask_cnic, normalize_cnic
from .ocr import run_ocr
from .ratelimit import RateLimiter
from .supabase_io import download_cnic_object, get_user_id_from_token, update_profile_ocr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("cnic-ocr")

app = FastAPI(title="Hi-Flow CNIC OCR", version="2.1.0")

if config.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

_gate_limiter = RateLimiter(config.GATE_RATE_LIMIT, config.GATE_RATE_PERIOD)

# Every OCR call runs on ONE dedicated worker thread. PaddlePaddle's inference
# predictor is not thread-safe — concurrent native calls segfault the process —
# and it is happiest called from a single stable thread. asyncio.to_thread()
# would otherwise fan calls out across the default executor. The 25s wait_for
# timeout around each submission is unchanged.
_ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")


async def _run_ocr_guarded(image_bytes: bytes):
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(_ocr_executor, run_ocr, image_bytes),
        timeout=config.OCR_TIMEOUT_SECONDS,
    )


def _client_key(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _strip_data_uri(value: str) -> str:
    """Bare base64 whether the client sent ``data:image/jpeg;base64,XXXX`` or ``XXXX``."""
    s = (value or "").strip()
    if s.startswith("data:") and "," in s:
        return s.split(",", 1)[1]
    return s


@app.exception_handler(RequestValidationError)
async def _on_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # FastAPI's default handler returns this body but logs NOTHING, so a 422 in
    # the Railway logs was previously undiagnosable. Log which field failed and
    # why — but SCRUB the offending value ("input"), which for this endpoint is
    # the CNIC number or the full base64 image and must never reach the logs.
    safe = [
        {"type": e.get("type"), "loc": e.get("loc"), "msg": e.get("msg")}
        for e in exc.errors()
    ]
    log.warning("422 on %s — request body failed schema validation: %s", request.url.path, safe)
    # Response body is byte-for-byte FastAPI's default (jsonable_encoder handles
    # non-serialisable `input` values) so existing clients keep working.
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


# ---------------------------------------------------------------------------
# 1. The registration gate — unauthenticated, stateless.
# ---------------------------------------------------------------------------
class GateRequest(BaseModel):
    # Deliberately permissive at the schema layer: any two strings parse, so a
    # 422 from this endpoint now means only one thing — the JSON body itself was
    # malformed / missing a field. Real validation (13-digit CNIC, a decodable
    # image) runs in the handler below and returns 400, not 422.
    entered_cnic: str
    image_base64: str


class GateResponse(BaseModel):
    status: str  # matched | mismatch | not_detected
    match: bool
    detected_cnic_masked: str
    confidence: float | None


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "service": "cnic-ocr", "version": "2.1.0"}


@app.post("/verify-cnic", response_model=GateResponse)
async def verify_cnic_gate(payload: GateRequest, request: Request) -> GateResponse:
    config.assert_ready_gate()

    if not _gate_limiter.allow(_client_key(request)):
        raise HTTPException(429, "Too many verification attempts. Please wait a minute and try again.")

    # entered_cnic — accept "35201-1234567-1" or "3520112345671" alike
    # (is_valid_cnic / compare both normalize to digits internally). A bad CNIC
    # is an application-level problem, NOT a malformed request -> 400, not 422.
    if not is_valid_cnic(payload.entered_cnic):
        raise HTTPException(400, "entered_cnic must contain exactly 13 digits")

    # image_base64 — accept a bare base64 string OR a full "data:image/...;base64,XXX" URI
    raw_field = payload.image_base64 or ""
    image_b64 = _strip_data_uri(raw_field)
    had_data_uri = image_b64 != raw_field.strip()
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "image_base64 is not valid base64")

    if len(raw) < 512:
        raise HTTPException(400, "CNIC image is empty or unreadable")
    if len(raw) > config.MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, "CNIC image too large")

    # Diagnostic only: lengths + payload shape. Never the CNIC, never the image.
    log.info(
        "gate request accepted: entered_len=%d normalized_len=%d image_b64_len=%d "
        "data_uri_prefix=%s image_bytes=%d",
        len(payload.entered_cnic or ""),
        len(normalize_cnic(payload.entered_cnic)),
        len(raw_field),
        had_data_uri,
        len(raw),
    )

    try:
        result = await _run_ocr_guarded(raw)
    except asyncio.TimeoutError:
        # A processing timeout is a server-side condition, not a bad request body.
        raise HTTPException(504, "Could not read the CNIC image in time")
    except ValueError:
        raise HTTPException(400, "Invalid or corrupt image")
    except Exception:  # noqa: BLE001
        log.exception("gate OCR crashed")
        raise HTTPException(500, "OCR processing failed")

    status, detected = compare(payload.entered_cnic, result.candidates)
    if status == "ocr_failed":
        status = "not_detected"

    log.info(
        "gate status=%s detected=%s conf=%s lines=%d",
        status,
        mask_cnic(detected),
        result.confidence,
        result.line_count,
    )
    return GateResponse(
        status=status,
        match=(status == "matched"),
        detected_cnic_masked=mask_cnic(detected),
        confidence=result.confidence if status == "matched" else None,
    )


# ---------------------------------------------------------------------------
# 2. Persist the result for the admin queue — authenticated.
# ---------------------------------------------------------------------------
class CommitRequest(BaseModel):
    entered_cnic: str = Field(min_length=5, max_length=40)
    front_path: str = Field(min_length=3)


@app.post("/verify-cnic/commit")
async def verify_cnic_commit(
    payload: CommitRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    config.assert_ready()

    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    caller_id = await get_user_id_from_token(token)
    if not caller_id:
        raise HTTPException(401, "Invalid or missing access token")

    norm_path = payload.front_path.lstrip("/")
    if not norm_path.startswith(f"{caller_id}/"):
        raise HTTPException(403, "front_path is outside the caller's folder")

    checked_at = datetime.now(timezone.utc).isoformat()

    try:
        image_bytes = await download_cnic_object(norm_path)
        result = await _run_ocr_guarded(image_bytes)
    except Exception as e:  # noqa: BLE001
        # Infrastructure failure (storage download / timeout / crash) — NOT an
        # OCR "no CNIC" result. The gate already passed; leave cnic_ocr_status
        # at 'pending' rather than writing a misleading 'ocr_failed'.
        log.warning("commit re-OCR infra failure for %s: %s", _mask_user(caller_id), e)
        return {"ok": True, "status": "pending"}

    status, detected = compare(payload.entered_cnic, result.candidates)
    conf = result.confidence if status != "ocr_failed" else None
    await _safe_persist(caller_id, status, detected, conf, checked_at)
    return {"ok": True, "status": status}


async def _safe_persist(
    user_id: str, status: str, detected: str | None, conf: float | None, checked_at: str
) -> None:
    try:
        await update_profile_ocr(
            user_id,
            status=status,
            detected_number=detected,
            confidence=conf,
            checked_at=checked_at,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("profile OCR write failed for %s: %s", _mask_user(user_id), e)


def _mask_user(uid: str) -> str:
    return f"{uid[:8]}…" if len(uid) > 8 else uid
