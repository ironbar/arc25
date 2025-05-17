FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# This has been copied from https://github.com/ironbar/insightface/blob/master/recognition/arcface_torch/docker/Dockerfile

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
# Add the deadsnakes PPA to get Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    curl \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# Default user for calculon cluster
RUN useradd -ms /bin/bash -u 1000 python-dev && \
    echo "python-dev ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers
# Default configuration
WORKDIR /home/python-dev
USER 1000