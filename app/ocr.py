"""OCR wrapper — RapidOCR on ONNX Runtime (CPU).

Replaces the previous PaddleOCR / PaddlePaddle engine, whose native runtime
SIGSEGV'd during inference on the deployment container (works on a normal
host, crash-loops on Railway; unresolved upstream across paddle 2.6.x–3.x).

RapidOCR runs the same PP-OCRv4 detection / classification / recognition
models through ONNX Runtime, which is CPU-stable in a slim container and has
no AVX / libstdc++ / oneDNN fragility. The models ship inside the wheel, so
there is no first-request download.

The public contract is unchanged:

    run_ocr(image_bytes: bytes) -> OcrResult(candidates, confidence, line_count)

so app/main.py, the CNIC extraction / matching logic, and the /verify-cnic
response are all untouched.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np

from .cnic import extract_cnic_candidates, normalize_cnic

log = logging.getLogger("cnic-ocr.ocr")

# ONNX Runtime otherwise sizes its intra-op pool to the host core count and
# pins every CPU during inference, starving the uvicorn event loop on a small
# (2 vCPU) container — new connections and Railway's own probes then stall and
# the edge returns a 502. Cap it so OCR always leaves a core for the app.
_INTRA_OP_THREADS = max(1, int(os.getenv("OCR_INTRA_OP_THREADS", "1")))

_ocr_lock = threading.Lock()      # single-flights engine construction
_infer_lock = threading.Lock()    # serialises inference calls
_ocr = None


def _get_ocr():
    """Lazily build the RapidOCR engine once and reuse it."""
    global _ocr
    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                from rapidocr_onnxruntime import RapidOCR  # heavy import — defer

                log.info(
                    "Loading RapidOCR (ONNX Runtime, CPU, intra_op=%d)…",
                    _INTRA_OP_THREADS,
                )
                _ocr = RapidOCR(
                    intra_op_num_threads=_INTRA_OP_THREADS,
                    inter_op_num_threads=1,
                )
                log.info("RapidOCR ready.")
    return _ocr


def _decode_image(data: bytes) -> "np.ndarray":
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image data")
    return img


class OcrResult:
    __slots__ = ("candidates", "confidence", "line_count")

    def __init__(self, candidates: list[str], confidence: float | None, line_count: int):
        self.candidates = candidates
        self.confidence = confidence
        self.line_count = line_count


def run_ocr(image_bytes: bytes) -> OcrResult:
    """Extract CNIC-looking numbers + a representative confidence from an image.

    Never raises for a "no text found" case — that is a normal `ocr_failed`
    outcome, returned as an empty candidate list.
    """
    img = _decode_image(image_bytes)
    engine = _get_ocr()

    # RapidOCR __call__ -> (result, elapse); result is
    #   [[box, text, score], ...]  or  None when nothing is detected.
    with _infer_lock:
        out = engine(img)
    raw = out[0] if isinstance(out, tuple) else out

    lines: list[tuple[str, float]] = []
    for entry in raw or []:
        try:
            text, conf = str(entry[1]), float(entry[2])
        except (IndexError, TypeError, ValueError):
            continue
        lines.append((text, conf))

    if not lines:
        log.info("ocr: 0 text regions detected")
        return OcrResult([], None, 0)

    joined = " ".join(text for text, _ in lines)
    candidates = extract_cnic_candidates(joined)

    # Confidence of the line that actually carried a CNIC, if we can pin it;
    # otherwise the mean over all lines. This is an OCR legibility signal
    # ONLY — never an authenticity score.
    conf: float | None = None
    if candidates:
        for text, c in lines:
            n = normalize_cnic(text)
            if n and n in candidates:
                conf = c
                break
    if conf is None:
        conf = sum(c for _, c in lines) / len(lines)

    log.info(
        "ocr: %d text regions, %d CNIC candidate(s)", len(lines), len(candidates)
    )
    return OcrResult(candidates, round(conf, 3), len(lines))
