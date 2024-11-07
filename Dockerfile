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
ENV CUDA_MODULE_LOADING LAZY
ENV LOG_VERBOSE 0

# TODO: Install the requirements file

# TODO: Copy the source code

# TODO: Run the server
