#!/bin/bash -ex

####### BASE IMAGES #######

# CUDA 11.3.1
docker build -t cuda113.x86_64 -f docker/Dockerfile.cuda113.x86_64.deps docker
# CUDA 11.8.0
docker build -t cuda118.x86_64 -f docker/Dockerfile.cuda118.x86_64.deps docker
# CUDA 12.1.1
docker build -t cuda121.x86_64 -f docker/Dockerfile.cuda121.x86_64.deps docker

# GCC 9 (minimum supported)
docker build -t nvimgcodecs_deps.x86_64.gcc9 -f docker/Dockerfile.x86_64.gcc9.deps docker
# GCC 10
docker build -t nvimgcodecs_deps.x86_64 -f docker/Dockerfile.x86_64.deps docker

####### BUILDER IMAGES #######

# GCC 9, CUDA 11.3 (minimum supported)
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-113-gcc9-v3" \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=nvimgcodecs_deps.x86_64.gcc9" \
    --build-arg "CUDA_IMAGE=cuda113.x86_64" \
    docker

# GCC 10, CUDA 11.8
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-118-v3" \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=nvimgcodecs_deps.x86_64" \
    --build-arg "CUDA_IMAGE=cuda118.x86_64" \
    docker

# GCC 10, CUDA 12.1
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-121-v3" \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=nvimgcodecs_deps.x86_64" \
    --build-arg "CUDA_IMAGE=cuda121.x86_64" \
    docker

####### TEST IMAGES #######

# CUDA 11.3 (minimum supported)
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-113-v3" \
     -f docker/Dockerfile \
     --build-arg "BASE=nvidia/cuda:11.3.1-runtime-ubuntu20.04" \
     --build-arg "VER_CUDA=11.3.1" \
     --build-arg "VER_UBUNTU=20.04" \
     docker

# CUDA 11.8
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-118-v3" \
     -f docker/Dockerfile \
     --build-arg "BASE=nvidia/cuda:11.8.0-runtime-ubuntu20.04" \
     --build-arg "VER_CUDA=11.8.0" \
     --build-arg "VER_UBUNTU=20.04" \
     docker

# CUDA 12.1
docker build -t "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-121-v3" \
     -f docker/Dockerfile \
     --build-arg "BASE=nvidia/cuda:12.1.1-runtime-ubuntu20.04" \
     --build-arg "VER_CUDA=12.1.1" \
     --build-arg "VER_UBUNTU=20.04" \
     docker
