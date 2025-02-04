FROM python:3.10-slim

# Install system dependencies (including build tools)
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    build-essential \
    python3-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Optionally set CMAKE_BUILD_PARALLEL_LEVEL to 1 to reduce memory usage during build
ENV CMAKE_BUILD_PARALLEL_LEVEL=1

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app"]
