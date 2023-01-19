FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Install Python3.9
RUN apt-get update \
  && apt-get install -y software-properties-common \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y \
  python3.9-dev \
  python3.9-venv \
  python3.9-distutils \
  git \
  curl \
  llvm \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3

# Install requirements
WORKDIR /
COPY requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copying the scripts
COPY run_hybrid_deberta.py run_hybrid_roberta.py input.jsonl/
RUN chmod +x run_hybrid_deberta.py
RUN chmod +x run_hybrid_roberta.py
ENTRYPOINT [ "/run_hybrid_deberta.py" ]

# NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/studwangsadirdja-spirex:0.0.1