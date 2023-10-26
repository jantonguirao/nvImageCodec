#!/bin/bash -ex

####### BUILDER IMAGES #######

# GCC 9 (minimum supported), CUDA 11.8
 docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-11.8-gcc9-v5"

# GCC 10, CUDA 11.8
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-11.8-v5"

# GCC 10, CUDA 12.3
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-x86_64:cuda-12.3-v5"

####### TEST IMAGES #######

# CUDA 11.3 (minimum supported)
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-11.3-v5"

# CUDA 11.8
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-11.8-v5"

# CUDA 12.3
docker push "gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-x86_64:cuda-12.1-v5"
