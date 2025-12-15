# ===== Base image ARM64 =====
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ===== System dependencies =====
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    v4l-utils \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ===== Upgrade pip =====
RUN python3 -m pip install --upgrade pip

# ===== Install RKNN Toolkit (ARM) =====
# ⚠️ Phải dùng bản tương thích RK3588
RUN pip install rknn-toolkit2==1.5.0

# ===== Install Python deps =====
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ===== Workdir =====
WORKDIR /app

# ===== Copy source =====
COPY . /app

# ===== OpenCV camera =====
ENV DISPLAY=:0

# ===== Run =====
CMD ["python3", "main_.py"]
