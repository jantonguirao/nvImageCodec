#!/bin/bash -ex

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}

####### BASE IMAGES #######

# Manylinux2014 aarch64 with GCC 10
docker build -t manylinux2014_aarch64.gcc10 -f docker/Dockerfile.gcc10 \
    --build-arg "FROM_IMAGE_NAME=quay.io/pypa/manylinux2014_aarch64" \
    docker

# CUDA 12.1.1, aarch64
docker build -t cuda12.1-aarch64 \
    -f docker/Dockerfile.cuda121.aarch64.deps \
    --ssh default=$HOME/.ssh/id_rsa \
    docker

# GCC 10, aarch64
docker build -t nvimgcodec_deps-aarch64 -f docker/Dockerfile.deps \
    --build-arg "FROM_IMAGE_NAME=manylinux2014_aarch64.gcc10" \
    --build-arg "ARCH=aarch64" \
    docker

####### BUILDER IMAGES #######

# GCC 10, CUDA 12.1, aarch64
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-aarch64:cuda-12.1-v3" \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=nvimgcodec_deps-aarch64" \
    --build-arg "CUDA_IMAGE=cuda12.1-aarch64" \
    docker

####### TEST IMAGES #######

# TODO
