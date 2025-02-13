FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES=all
ARG PYTHON_VERSION=3.9

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# https://github.com/haosulab/ManiSkill/issues/9
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# install dependencies
# RUN pip install --upgrade mani-skill && pip cache purge
RUN pip install torch 

ARG CACHE_BUST=1
RUN pip install nvitop
RUN git clone -b main --single-branch https://github.com/dwaitbhatt/ManiSkill.git /workspace/ManiSkill
WORKDIR /workspace/ManiSkill
RUN pip install -e .

# For TD-MPC2
COPY ./examples/baselines/tdmpc2/requirements.txt /workspace/ManiSkill/examples/baselines/tdmpc2/requirements.txt
RUN pip install -r /workspace/ManiSkill/examples/baselines/tdmpc2/requirements.txt
RUN pip install tensorboard wandb tqdm jupyterlab nvitop

# download physx GPU binary via sapien
RUN python -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"

CMD ["bash"]