#!/bin/bash -ex

BUILDER_CUDA_VERSION="12.1"
BUILDER_CUDA_SHORT_VER=${BUILDER_CUDA_VERSION//./}
BUILDER_ARCH="x86_64" # x86_64 or arm64-sbsa
BUILDER_VERSION="1"

CUDA_IMAGE="cuda${BUILDER_CUDA_VERSION}.${BUILDER_ARCH}"
DEPS_IMAGE="nvimgcodecs_deps.${BUILDER_ARCH}"
BUILDER_IMAGE="gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-${BUILDER_ARCH}:cuda-${BUILDER_CUDA_VERSION}-v${BUILDER_VERSION}"
# Note: Use build_dockers.sh to produce the image if needed

docker run --runtime=nvidia --gpus=all --rm -it -v ${PWD}:/opt/src ${BUILDER_IMAGE} /bin/bash
