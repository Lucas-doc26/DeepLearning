FROM docker.io/tensorflow/tensorflow:2.15.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        python3.10 python3.10-venv python3.10-distutils \
        build-essential git wget curl nano sudo libgl1 \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
        libglib2.0-0 libsm6 libxext6 libxrender-dev\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/lucas/DeepLearning

COPY requirements.txt .

RUN python3.10 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

CMD ["bash"]