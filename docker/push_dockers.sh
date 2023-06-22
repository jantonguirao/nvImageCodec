#!/bin/bash -ex

BUILDER_ARCH="x86_64"
BUILDER_VERSION="1"

CUDA_VERSIONS=("11.8" "12.1")
for BUILDER_CUDA_VERSION in "${CUDA_VERSIONS[@]}"; do
    BUILDER_CUDA_SHORT_VER=${BUILDER_CUDA_VERSION//./}

    CUDA_IMAGE="cuda${BUILDER_CUDA_VERSION}.${BUILDER_ARCH}"
    DEPS_IMAGE="nvimgcodecs_deps.${BUILDER_ARCH}"
    BUILDER_IMAGE="gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-${BUILDER_ARCH}:cuda-${BUILDER_CUDA_VERSION}-v${BUILDER_VERSION}"

    docker push ${BUILDER_IMAGE}
done