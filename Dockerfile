# CNIC OCR microservice
#
# Base image: the FULL python:3.11 (Debian bookworm), NOT python:3.11-slim.
# The pinned stack (paddlepaddle 2.6.2 / paddleocr 2.9.1) runs cleanly on a
# normal x86-64 host but the PaddlePaddle native runtime SIGSEGVs on the slim
# image — slim omits shared libraries / a consistent libstdc++/libgomp that
# Paddle's manylinux wheel links against at runtime. The full image ships them.
FROM python:3.11

# OpenCV runtime deps (present on full image, kept explicit for clarity).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Keep native math/threading pools modest — the container is CPU-limited, and
# oneDNN (MKL-DNN) has known native crashes in paddlepaddle 2.6.x, so disable it.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    FLAGS_use_mkldnn=0

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

# Warm the PaddleOCR weights into the image so the first request is fast.
# (Safe to remove if your build environment has no network access — the
#  model then downloads lazily on the first /verify-cnic call instead.)
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', show_log=False)" || true

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
