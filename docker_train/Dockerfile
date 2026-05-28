# =============================================================================
# ODIN Training Image — NBV Stage3 Isaac Dataset
# =============================================================================
# Base: PyTorch 2.2.0 + CUDA 11.8 (compatible with driver 520+ / server has 525.x)
#
# Build once (~30-40 min), run many times:
#   docker-compose build
#   docker-compose up -d nbv-train
# =============================================================================

FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Avoid interactive prompts from apt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libxi6 \
    libxmu6 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# =============================================================================
# 1. Clone ODIN repository
# =============================================================================
ARG ODIN_BRANCH=feature/nbv_dataset_process
ARG ODIN_REPO=https://github.com/SergKurchev/my_odin.git

RUN echo "Cloning ODIN branch: ${ODIN_BRANCH}" && \
    git clone -q -b "${ODIN_BRANCH}" "${ODIN_REPO}" odin

WORKDIR /workspace/odin

# =============================================================================
# 2. Python dependencies (mirrors Kaggle notebook install order)
# =============================================================================

# Step 2.1 — torch-scatter (must match PyTorch 2.2.0 + cu118)
RUN pip install --no-cache-dir torch-scatter \
    -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# Step 2.2 — NumPy < 2 first (avoids conflicts during later installs)
RUN pip install --no-cache-dir "numpy<2" --force-reinstall

# Step 2.3 — Build tools + modern COCO API
RUN pip install --no-cache-dir cython setuptools wheel pycocotools

# Step 2.4 — Patch requirements.txt:
#   - Remove packages installed separately (detectron2, pytorch3d, waspinator, ai2thor, prior)
#   - Upgrade outdated pinned versions (pyyaml 5.3.1, Pillow 9.1.0)
COPY docker_train/patch_requirements.py /tmp/patch_requirements.py
RUN python3 /tmp/patch_requirements.py

# Step 2.5 — ODIN requirements + extra build tools
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ninja fvcore iopath

# Step 2.6 — Detectron2 (Facebook Research, no-build-isolation required)
RUN pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git"

# Step 2.7 — PyTorch3D with CUDA
#   TORCH_CUDA_ARCH_LIST covers:
#     7.0 = V100          7.5 = T4 / RTX 20xx
#     8.0 = A100          8.6 = RTX 30xx / A5000
#     8.9 = RTX 40xx (Ada Lovelace) ← your GPU
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"

RUN pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git"

# Step 2.8 — Pin final NumPy + OpenCV versions
#   (must come AFTER pytorch3d which may pull a newer numpy)
RUN pip install --no-cache-dir numpy==1.26.4 --force-reinstall && \
    pip install --no-cache-dir opencv-python-headless==4.8.0.76

# =============================================================================
# 3. Compile CUDA kernels
# =============================================================================

# 3.1 — pointops2: KNN search and aggregation for 3D point clouds
RUN cd libs/pointops2 && \
    rm -rf build && \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9" \
    python setup.py install && \
    cd /workspace/odin

# 3.2 — Deformable Attention ops (pixel decoder)
RUN cd /workspace/odin/odin/modeling/pixel_decoder/ops && \
    rm -rf build && \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9" \
    python setup.py build install && \
    cd /workspace/odin

# =============================================================================
# 4. RL / NBV Project dependencies
# =============================================================================
RUN pip install --no-cache-dir \
    pybullet \
    gymnasium \
    "imageio>=2.9" \
    imageio[ffmpeg] \
    pynvml \
    pandas \
    matplotlib

# =============================================================================
# 5. Environment and entrypoint
# =============================================================================

# Copy project source into separate directory
COPY src /workspace/nbv_project/src
COPY train_odin_sac_rl.py \
     evaluate.py \
     config.py \
     /workspace/nbv_project/

# PYTHONPATH must include both ODIN and the project root
ENV PYTHONPATH=/workspace/odin:/workspace/nbv_project

WORKDIR /workspace/nbv_project

# Default entrypoint (overridden in docker-compose per service)
ENTRYPOINT ["/bin/bash"]
