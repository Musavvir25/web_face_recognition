# Use a lightweight Python 3.10 image
FROM python:3.10-slim

# Install system dependencies including build tools and libgl1-mesa-glx for OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    build-essential \
    python3-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Optionally, reduce build parallelism to lower memory usage during package builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=1

# Upgrade pip and install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Define the command to run your application using Gunicorn
CMD ["gunicorn", "app:app"]
