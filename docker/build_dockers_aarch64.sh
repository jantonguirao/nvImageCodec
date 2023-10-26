#!/bin/bash -ex

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-0}

####### BASE IMAGES #######

# Manylinux2014 aarch64 with GCC 10
docker build -t manylinux2014_aarch64.gcc10 -f docker/Dockerfile.gcc10 \
    --build-arg "FROM_IMAGE_NAME=quay.io/pypa/manylinux2014_aarch64" \
    docker

# CUDA 11.8, aarch64
docker build -t cuda11.8-aarch64 \
    -f docker/Dockerfile.cuda118.aarch64.deps \
    docker

# CUDA 12.3, aarch64
docker build -t cuda12.3-aarch64 \
    -f docker/Dockerfile.cuda123.aarch64.deps \
    docker

# GCC 10, aarch64
docker build -t nvimgcodec_deps-aarch64 -f docker/Dockerfile.deps \
    --build-arg "FROM_IMAGE_NAME=manylinux2014_aarch64.gcc10" \
    --build-arg "ARCH=aarch64" \
    docker

####### BUILDER IMAGES #######

# GCC 10, CUDA 11.8, aarch64
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-aarch64:cuda-11.8-v4" \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=nvimgcodec_deps-aarch64" \
    --build-arg "CUDA_IMAGE=cuda11.8-aarch64" \
    docker

# GCC 10, CUDA 12.3, aarch64
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-aarch64:cuda-12.3-v4" \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=nvimgcodec_deps-aarch64" \
    --build-arg "CUDA_IMAGE=cuda12.3-aarch64" \
    docker

####### TEST IMAGES #######

# CUDA 11.8
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-aarch64:cuda-11.8-v4" \
     -f docker/Dockerfile.aarch64 \
     --build-arg "BASE=nvidia/cuda:11.8.0-runtime-ubuntu20.04" \
     --build-arg "VER_CUDA=11.8.0" \
     --build-arg "VER_UBUNTU=20.04" \
     docker
