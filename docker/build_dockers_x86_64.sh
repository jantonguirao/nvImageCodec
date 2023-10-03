#!/bin/bash -ex

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-0}

####### BASE IMAGES #######

# Manylinux2014 x86_64 with GCC 9
docker build -t "manylinux2014_x86_64.gcc9" -f docker/Dockerfile.gcc9 \
    --build-arg "FROM_IMAGE_NAME=quay.io/pypa/manylinux2014_x86_64" \
    docker

# Manylinux2014 x86_64 with GCC 10
docker build -t "manylinux2014_x86_64.gcc10" -f docker/Dockerfile.gcc10 \
    --build-arg "FROM_IMAGE_NAME=quay.io/pypa/manylinux2014_x86_64" \
    docker

# CUDA 11.8.0, x86_64
docker build -t "cuda11.8-x86_64" -f docker/Dockerfile.cuda118.x86_64.deps docker
# CUDA 12.1.1, x86_64
docker build -t "cuda12.1-x86_64" -f docker/Dockerfile.cuda121.x86_64.deps docker

# GCC 9 (minimum supported), x86_64
docker build -t "nvimgcodec_deps-x86_64-gcc9" -f docker/Dockerfile.deps \
    --build-arg "FROM_IMAGE_NAME=manylinux2014_x86_64.gcc9" \
    docker

# GCC 10, x86_64
docker build -t "nvimgcodec_deps-x86_64" -f docker/Dockerfile.deps \
    --build-arg "FROM_IMAGE_NAME=manylinux2014_x86_64.gcc10" \
    docker

####### BUILDER IMAGES #######

# GCC 9 (minimum supported), CUDA 11.8, x86_64
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-11.8-gcc9-v3" \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=nvimgcodec_deps-x86_64-gcc9" \
    --build-arg "CUDA_IMAGE=cuda11.8-x86_64" \
    docker

# GCC 10, CUDA 11.8, x86_64
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-11.8-v3" \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=nvimgcodec_deps-x86_64" \
    --build-arg "CUDA_IMAGE=cuda11.8-x86_64" \
    docker

# GCC 10, CUDA 12.1, x86_64
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-12.1-v3" \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=nvimgcodec_deps-x86_64" \
    --build-arg "CUDA_IMAGE=cuda12.1-x86_64" \
    docker

####### TEST IMAGES #######

# CUDA 11.3 (minimum supported)
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-11.3-v4" \
     -f docker/Dockerfile \
     --build-arg "BASE=nvidia/cuda:11.3.1-runtime-ubuntu20.04" \
     --build-arg "VER_CUDA=11.3.1" \
     --build-arg "VER_UBUNTU=20.04" \
     docker

# CUDA 11.8
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-11.8-v4" \
     -f docker/Dockerfile \
     --build-arg "BASE=nvidia/cuda:11.8.0-runtime-ubuntu20.04" \
     --build-arg "VER_CUDA=11.8.0" \
     --build-arg "VER_UBUNTU=20.04" \
     docker

# CUDA 12.1
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-12.1-v4" \
     -f docker/Dockerfile \
     --build-arg "BASE=nvidia/cuda:12.1.1-runtime-ubuntu20.04" \
     --build-arg "VER_CUDA=12.1.1" \
     --build-arg "VER_UBUNTU=20.04" \
     docker
