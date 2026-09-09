# CNIC OCR microservice
FROM python:3.11-slim

# PaddleOCR / OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Constrain native thread pools: the container has a hard ~2 vCPU / ~1 GB cgroup
# limit, but PaddlePaddle / OpenBLAS / OpenMP otherwise size their pools to the
# host core count and oversubscribe — stalling inference past the request
# timeout and inflating memory until the worker dies (504 -> 502).
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    FLAGS_use_mkldnn=0 \
    OCR_CPU_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

# Warm the PaddleOCR weights into the image using the SAME runtime config, so
# the first request never has to download + build all three models under the
# container's memory limit. Non-fatal: with no network at build time the model
# downloads lazily on the first /verify-cnic call instead.
RUN python -c "from app.ocr import _get_ocr; _get_ocr()" || true
# Surface in the build log whether the weights actually baked in.
RUN python -c "import os; p=os.path.expanduser('~/.paddleocr'); print('[build] paddleocr weights:', 'present at '+p if os.path.isdir(p) else 'MISSING — will download on first request')"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
