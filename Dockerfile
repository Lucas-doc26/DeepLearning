FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


WORKDIR /workspace

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip python3.10-venv \
      build-essential git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Instala TensorFlow GPU 2.15
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
CMD ["/bin/bash"]

