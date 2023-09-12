#!/bin/bash -ex

####### BUILDER IMAGES #######

# GCC 9, CUDA 11.3 (minimum supported)
 docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-11.3-gcc9-v3"

# GCC 10, CUDA 11.8
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-11.8-v3"

# GCC 10, CUDA 12.1
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-12.1-v3"

####### TEST IMAGES #######

# CUDA 11.3 (minimum supported)
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-11.3-v3"

# CUDA 11.8
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-11.8-v3"

# CUDA 12.1
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-12.1-v3"
