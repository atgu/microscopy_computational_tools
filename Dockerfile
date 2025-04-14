FROM 'debian:12-slim'
ARG JXL_VERSION="0.11.1"

RUN apt update && \
    # tools: python3, python3-pip, git, curl, moreutils (has parallel)
    # dependencies for Pillow, opencv and libjxl: libtiff-dev zlib1g-dev libhwy-dev libgl1 libglib2.0-0
    apt install -y python3 python3-pip git curl moreutils && \
    rm /usr/lib/python*/EXTERNALLY-MANAGED && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    # torch in the debian12 repo is too old for cellpose
    # CP-CNN requires efficientnet and tensorflow
    # install all packages via pip (vs apt) since pip does not install torch if an outdated sympy is installed by apt
    # pinning to TF 2.17 due to https://github.com/tensorflow/tensorflow/issues/78784 (current version is 2.18.0)
    pip3 install --no-cache-dir torch pandas numpy scipy numba efficientnet tensorflow==2.17.1 tensorflow-cpu==2.17.1 pillow pillow-jxl-plugin cellpose && \
    rm -rf /var/lib/apt/lists/*

# the nvidia-container-toolkit allows using the GPU on Hail Batch container without installing the full driver
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt update  && \
    apt install -y nvidia-container-toolkit && \
    rm -rf /var/lib/apt/lists/*

COPY cellpose /scripts/cellpose
COPY embeddings /scripts/embeddings