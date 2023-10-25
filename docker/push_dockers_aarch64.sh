#!/bin/bash -ex

####### BUILDER IMAGES #######

# GCC 10, CUDA 12.1
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-aarch64:cuda-12.3-v4"

####### TEST IMAGES #######

# TODO