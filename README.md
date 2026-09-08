# Hi-Flow CNIC OCR service

A small, isolated FastAPI service that runs **PaddleOCR** over a CNIC image
and reports whether the number it reads matches the number the user typed.

It is deliberately dumb: it **never approves or rejects** an account and
**never claims the CNIC is genuine / government-issued**. It only does text
detection + number comparison. The admin panel remains the sole authority
for `verification_status`.

> **This service is now a HARD DEPENDENCY for new registration.** The mobile
> app calls `/verify-cnic` *before* it creates the auth user and refuses to
> continue unless the response is `matched`. If the service is down,
> unreachable, or `EXPO_PUBLIC_CNIC_OCR_URL` is unset, **new Customer /
> Shopkeeper signups are blocked** (fail-closed, by design). Existing users
> are unaffected — login / ordering / everything else never calls this.

```
mobile signup
   │ enter CNIC + pick front image
   ▼
POST /verify-cnic  { entered_cnic, image_base64 }        ← unauthenticated, stateless
   │ PaddleOCR → extract 13-digit CNIC → normalize → compare
   ▼
   status = matched ?  ── no ──▶  registration ABORTS, no account created
   │ yes
   ▼
supabase.auth.signUp()  →  upload images to private bucket cnic-documents
   ▼
POST /verify-cnic/commit  { entered_cnic, front_path }   ← authenticated (user JWT)
   │ service-role: download stored image, re-OCR, write profiles.cnic_ocr_*
   ▼
admin queue shows the OCR result — admin still verifies manually
```

## Endpoints

### `POST /verify-cnic`  — the registration gate (unauthenticated)

```jsonc
// request
{ "entered_cnic": "3520212345671", "image_base64": "<base64 JPEG>" }

// 200
{
  "status": "matched",              // matched | mismatch | not_detected
  "match": true,
  "detected_cnic_masked": "*****-*****67-1",
  "confidence": 0.94                 // OCR legibility 0–1, NOT authenticity; null unless matched
}
```

- `400` corrupt/blank image · `413` too large · `422` unreadable / no CNIC found
  · `429` rate-limited · `500` OCR crashed.
- The mobile client treats **anything except `200 {match:true, status:"matched"}`**
  as "block registration".
- Writes nothing, stores nothing, needs no Supabase credentials.
- Rate-limited in-process (`GATE_RATE_LIMIT` calls per `GATE_RATE_PERIOD_SECONDS`
  per client IP / `X-Forwarded-For`). Put a real edge rate-limit in front too.

### `POST /verify-cnic/commit`  — persist for the admin queue (authenticated)

Header: `Authorization: Bearer <the signing-up user's Supabase access token>`

```jsonc
{ "entered_cnic": "3520212345671", "front_path": "<user_id>/cnic_front_1730000000000.jpg" }
// 200 -> { "ok": true, "status": "matched" }
```

- `401` bad token · `403` `front_path` not under the caller's own folder.
- Downloads the stored image with the **service role**, re-runs OCR, writes
  `profiles.cnic_ocr_status / _detected_number / _confidence / _checked_at`.
- Best-effort from the app's side — the gate already passed; if this fails
  the admin queue just shows `cnic_ocr_status = pending`.

### `GET /health` → `{ "ok": true }`

## Run locally

```bash
cd backend/cnic-ocr
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # fill SUPABASE_SERVICE_ROLE_KEY (commit endpoint only)
uvicorn app.main:app --reload --port 8000
python -m pytest                                      # pure-logic tests, no model needed
```

> **Paddle install trouble?** `paddlepaddle` wheels are platform-specific.
> If `pip install -r requirements.txt` fails, install the wheel for your
> OS/arch from <https://www.paddlepaddle.org.cn/en> first, then re-run.

## Deploy (Docker)

```bash
docker build -t hiflow-cnic-ocr backend/cnic-ocr
docker run -p 8000:8000 --env-file backend/cnic-ocr/.env hiflow-cnic-ocr
```

Runs on Railway / Render / Fly.io / any container host. Set the env vars
from `.env.example`. Give it **≥1 GB RAM**. The image pre-downloads the
PaddleOCR weights at build so the first request is fast.

**Deploy this and set `EXPO_PUBLIC_CNIC_OCR_URL` in `hi-flow-mobile/.env`
before shipping a build — without it, nobody can register.**

## Security

- The service-role key lives **only** here. Never sent to the phone, no
  `EXPO_PUBLIC_*` counterpart. The `/verify-cnic` gate needs no credentials
  at all.
- CNIC images stay in the private `cnic-documents` bucket. The gate receives
  the image in-request and does not persist it; `/commit` downloads with the
  service role and does not persist it.
- Full CNIC numbers are never logged or returned — only a masked form.
- `cnic_ocr_confidence` is an OCR-legibility number, **not** an
  authenticity / genuineness percentage.
