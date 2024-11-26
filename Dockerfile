# Description: This file is used to build the docker image for the model server
FROM nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python-is-python3 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    wget \
    google-perftools \
    && rm -rf /var/lib/apt/lists/*
RUN echo "LD_PRELOAD=/usr/lib/libtcmalloc.so.4" | tee -a /etc/environment
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN sed -i 's/python3/python3.11/g' /usr/bin/pip3
RUN sed -i 's/python3/python3.11/g' /usr/bin/pip

# Set up the workspace
WORKDIR /workspace/
# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_MODULE_LOADING=LAZY
ENV LOG_VERBOSE=0
# Set up (AI) model name
ENV MODEL_NAME="segmind/tiny-sd"
ENV WARMUP_MODEL="true"

# Copy current directory to Docker workspace
COPY . /workspace/

# Install torch first to avoid conflicts with other packages.
RUN pip install torch==2.3.0
# Install dependencies
RUN pip install -r requirements.txt
#
## Run the model server
CMD ["python3", "main.py"]
