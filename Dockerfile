# KalMamba environment for Isambard-AI (linux/arm64, NVIDIA GH200).
# Build on a compute node:  sbatch --wait scripts/isambard/build_image.sbatch
# Run:                      srun --gpus=1 podman-hpc run --rm --gpu -v "$PWD:/opt/kalmamba" -w /opt/kalmamba kalmamba:v1 <command>
#
# The code is mounted at run time instead of copied, so code changes need no rebuild.
FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04

RUN apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update \
    && apt-get install --assume-yes --no-install-recommends git python3.10-dev python3.10-venv \
        libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# `podman-hpc run --gpu` mounts the host's libcuda into /usr/lib64, which Ubuntu does not search.
ENV LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN python3.10 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.7.1+cu128 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# mamba-ssm and causal-conv1d have no Arm64 wheels, so they are compiled here (GH200 is sm_90).
# They are built from the release tags because the PyPI source archives lack the C++ sources.
RUN pip install --no-cache-dir ninja packaging wheel setuptools \
    && export MAX_JOBS=32 TORCH_CUDA_ARCH_LIST=9.0 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
    && pip install --no-cache-dir --no-build-isolation \
        "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.0.post8" \
    && pip install --no-cache-dir --no-build-isolation \
        "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.4"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
