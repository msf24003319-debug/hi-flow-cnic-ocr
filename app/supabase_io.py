"""Thin Supabase REST/Storage client for this service.

Uses the SERVICE ROLE key, so every call here bypasses RLS — keep the
surface tiny and never expose it to the client.
"""
from __future__ import annotations

import logging

import httpx

from . import config

log = logging.getLogger("cnic-ocr.supabase")


def _svc_headers() -> dict[str, str]:
    return {
        "apikey": config.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {config.SUPABASE_SERVICE_ROLE_KEY}",
    }


async def get_user_id_from_token(access_token: str) -> str | None:
    """Resolve a mobile user's access token -> their auth user id.

    Returns None if the token is missing / invalid / expired.
    """
    if not access_token:
        return None
    url = f"{config.SUPABASE_URL}/auth/v1/user"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                url,
                headers={
                    "apikey": config.SUPABASE_SERVICE_ROLE_KEY,
                    "Authorization": f"Bearer {access_token}",
                },
            )
        if r.status_code != 200:
            log.info("token check -> %s", r.status_code)
            return None
        return r.json().get("id")
    except httpx.HTTPError as e:
        log.warning("token check failed: %s", e)
        return None


async def download_cnic_object(object_path: str) -> bytes:
    """Download a private cnic-documents object by its storage path."""
    path = object_path.lstrip("/")
    url = f"{config.SUPABASE_URL}/storage/v1/object/{config.CNIC_BUCKET}/{path}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=_svc_headers())
    r.raise_for_status()
    return r.content


async def update_profile_ocr(
    user_id: str,
    *,
    status: str,
    detected_number: str | None,
    confidence: float | None,
    checked_at: str,
) -> None:
    """PATCH profiles.cnic_ocr_* for one user. Best-effort — raises on HTTP error."""
    url = f"{config.SUPABASE_URL}/rest/v1/profiles?id=eq.{user_id}"
    body = {
        "cnic_ocr_status": status,
        "cnic_ocr_detected_number": detected_number,
        "cnic_ocr_confidence": confidence,
        "cnic_ocr_checked_at": checked_at,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.patch(
            url,
            json=body,
            headers={**_svc_headers(), "Content-Type": "application/json", "Prefer": "return=minimal"},
        )
    r.raise_for_status()
