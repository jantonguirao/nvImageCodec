#!/bin/bash -ex

BUILDER_CUDA_VERSION="11.8"
BUILDER_CUDA_SHORT_VER=${BUILDER_CUDA_VERSION//./}
BUILDER_ARCH="x86_64"
BUILDER_VERSION="1"

CUDA_IMAGE="cuda${BUILDER_CUDA_VERSION}.${BUILDER_ARCH}"
DEPS_IMAGE="nvimgcodec_deps.${BUILDER_ARCH}"
BUILDER_IMAGE="gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-${BUILDER_ARCH}:cuda-${BUILDER_CUDA_VERSION}-v${BUILDER_VERSION}"
# Note: Use build_dockers.sh to produce the image if needed

docker run --rm -it -v ${PWD}:/opt/src ${BUILDER_IMAGE} /bin/bash -c \
    'WHL_OUTDIR=/opt/src/artifacts && mkdir -p /opt/src/build_docker && cd /opt/src/build_docker && source ../docker/build_helper.sh'
