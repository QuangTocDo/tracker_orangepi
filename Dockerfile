# =========================
# Base image
# =========================
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =========================
# TOOLCHAIN cho dlib (BẮT BUỘC)
# =========================
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Workdir
# =========================
WORKDIR /app

# =========================
# Python deps
# =========================
COPY requirement.txt .

# ⚠️ ép build dlib không dùng GPU, không AVX
ENV CMAKE_ARGS="-DDLIB_USE_CUDA=0 -DDLIB_USE_AVX_INSTRUCTIONS=0"
ENV FORCE_CMAKE=1

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirement.txt

# =========================
# Copy source
# =========================
COPY . .

CMD ["python", "main_.py"]

