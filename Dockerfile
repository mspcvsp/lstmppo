FROM python:3.10-slim

# 1. System deps (rarely change)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python deps — cache this layer
COPY pyproject.toml .
RUN pip install --upgrade pip
RUN pip install .

# 3. Copy source code LAST so changes don’t bust cache
COPY . .

# Optional: install test deps only if you want tests inside container
# RUN pip install pytest