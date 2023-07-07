#!/bin/bash -ex

IMG_ARCH="x86_64"
IMG_VERSION="1"
IMG_UBUNTU="20.04"

CUDA_VERSIONS=("11.8.0" "12.1.1")
for IMG_CUDA_VERSION in "${CUDA_VERSIONS[@]}"; do
    BUILDER_IMAGE="gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-${IMG_ARCH}:cuda-${IMG_CUDA_VERSION}-v${IMG_VERSION}"
    docker push ${BUILDER_IMAGE}

    TEST_IMAGE="gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-${IMG_ARCH}:cuda-${IMG_CUDA_VERSION}-v${IMG_VERSION}"
    docker push ${TEST_IMAGE}
done
