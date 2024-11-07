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

WORKDIR /workspace/
# install requirements file
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
ENV DEBIAN_FRONTEND=noninteractive


COPY main.py .
COPY helpers helpers
# copy models IF models exists
RUN mkdir -p models
COPY models models
# copy scripts IF scripts exists
RUN mkdir -p scripts
COPY scripts scripts

ENV CUDA_MODULE_LOADING LAZY
ENV LOG_VERBOSE 0
ENV MODE "deployed"
COPY docker_scripts/run_server.sh .
CMD bash run_server.sh
