# CNIC OCR microservice — FastAPI + RapidOCR (ONNX Runtime, CPU).
FROM python:3.11-slim

# OpenCV native deps (opencv-python needs libGL / libglib); libgomp for
# ONNX Runtime's OpenMP thread pool.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    OMP_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

# Sanity-check the OCR engine at build time (models are bundled in the wheel,
# so this needs no network). Non-fatal.
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" || true

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
