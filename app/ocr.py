"""PaddleOCR wrapper.

The model is loaded once, lazily, on the first request (keeps container
start-up fast and avoids downloading weights during image build if the
network there is restricted).
"""
from __future__ import annotations

import io
import logging
import os
import threading

# --- native-runtime safety: MUST run before numpy / paddle / cv2 import ------
# The container runs under a hard cgroup CPU limit (~2 vCPU) and ~1 GB RAM.
# PaddlePaddle, OpenBLAS and OpenMP each otherwise size their thread pools to
# the *host* core count (dozens). On a constrained container that oversubscription
# makes a single inference stall past the request timeout AND inflates memory
# (per-thread arenas), which is what took the worker down mid-request
# (observed as 504 then 502). Pin every native pool to one thread.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# Quieten PaddlePaddle's native (GLOG) chatter before paddle is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("FLAGS_call_stack_level", "0")
# oneDNN/MKL-DNN has known native crashes in paddlepaddle 2.6.x — keep it off.
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import numpy as np

from . import config
from .cnic import extract_cnic_candidates

log = logging.getLogger("cnic-ocr.ocr")

# PaddleOCR defaults cpu_threads=10 *per predictor* (det + cls + rec). Far too
# many for a 2-vCPU container; 1 is stable. Override via OCR_CPU_THREADS only if
# the container is resized to more cores.
_OCR_CPU_THREADS = max(1, int(os.getenv("OCR_CPU_THREADS", "1")))

_ocr_lock = threading.Lock()      # guards one-time model construction
_infer_lock = threading.Lock()    # serialises native inference (predictor is NOT thread-safe)
_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                from paddleocr import PaddleOCR  # heavy import — defer

                log.info(
                    "Loading PaddleOCR model (lang=%s, cpu_threads=%d)…",
                    config.OCR_LANG,
                    _OCR_CPU_THREADS,
                )
                _ocr = PaddleOCR(
                    use_angle_cls=True,  # handles 90/180/270-rotated cards
                    lang=config.OCR_LANG,
                    show_log=False,
                    enable_mkldnn=False,      # oneDNN SIGSEGVs on paddle 2.6.x
                    cpu_threads=_OCR_CPU_THREADS,
                )
                log.info("PaddleOCR ready.")
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
    ocr = _get_ocr()

    # The predictor is not thread-safe: two concurrent native .ocr() calls
    # crash the process. Serialise them (callers already run on one executor
    # thread; this is the hard guarantee).
    with _infer_lock:
        raw = ocr.ocr(img, cls=True)
    # PaddleOCR returns [ [ [box, (text, conf)], ... ] ] — one entry per image.
    lines: list[tuple[str, float]] = []
    for page in raw or []:
        for entry in page or []:
            try:
                text, conf = entry[1][0], float(entry[1][1])
            except (IndexError, TypeError, ValueError):
                continue
            lines.append((text, conf))

    if not lines:
        return OcrResult([], None, 0)

    joined = " ".join(text for text, _ in lines)
    candidates = extract_cnic_candidates(joined)

    # Confidence of the line that actually carried a CNIC, if we can pin it;
    # otherwise the mean over all lines. This is an OCR legibility signal
    # ONLY — never an authenticity score.
    conf: float | None = None
    if candidates:
        from .cnic import normalize_cnic

        for text, c in lines:
            if normalize_cnic(text) and normalize_cnic(text) in candidates:
                conf = c
                break
    if conf is None:
        conf = sum(c for _, c in lines) / len(lines)

    return OcrResult(candidates, round(conf, 3), len(lines))
