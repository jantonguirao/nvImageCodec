#!/bin/bash -ex

####### BUILDER IMAGES #######

# GCC 10, CUDA 11.8
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-aarch64:cuda-11.8-v5"

# GCC 10, CUDA 12.3
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-aarch64:cuda-12.3-v5"

####### TEST IMAGES #######

# CUDA 11.8
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-aarch64:cuda-11.8-v5"
