#!/bin/bash -ex

IMG_ARCH="x86_64"
IMG_UBUNTU="20.04"

CUDA_VERSIONS=("11.8.0" "12.1.1")
for IMG_CUDA_VERSION in "${CUDA_VERSIONS[@]}"; do
    IMG_CUDA_VERSION_MAJOR_MINOR=$(echo $IMG_CUDA_VERSION | sed -E 's/^([0-9]+\.[0-9]+).*$/\1/')
    IMG_CUDA_SHORT_VER=${IMG_CUDA_VERSION_MAJOR_MINOR//./}
    TEST_IMG_BASE="nvidia/cuda:${IMG_CUDA_VERSION}-runtime-ubuntu${IMG_UBUNTU}"
    CUDA_IMAGE="cuda${IMG_CUDA_VERSION}.${IMG_ARCH}"
    DEPS_IMAGE="nvimgcodecs_deps.${IMG_ARCH}"

    BUILDER_IMAGE_VER="2"
    BUILDER_IMAGE="gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/build-linux-${IMG_ARCH}:cuda-${IMG_CUDA_VERSION}-v${BUILDER_IMAGE_VER}"

    TEST_IMAGE_VER="1"
    TEST_IMAGE="gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/runner-linux-${IMG_ARCH}:cuda-${IMG_CUDA_VERSION}-v${TEST_IMAGE_VER}"

    DOCS_IMAGE_VER="1"
    DOCS_IMAGE="gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/docs-linux-${IMG_ARCH}:cuda-${IMG_CUDA_VERSION}-v${DOCS_IMAGE_VER}"

    # Building dependencies
    docker build -t ${CUDA_IMAGE} -f docker/Dockerfile.cuda${IMG_CUDA_SHORT_VER}.${IMG_ARCH}.deps docker
    docker build -t ${DEPS_IMAGE} -f docker/Dockerfile.${IMG_ARCH}.deps docker

    docker build -t ${BUILDER_IMAGE} -f docker/Dockerfile.cuda.deps --build-arg "FROM_IMAGE_NAME=${DEPS_IMAGE}" --build-arg "CUDA_IMAGE=${CUDA_IMAGE}" docker

    docker build -t ${TEST_IMAGE} -f docker/Dockerfile --build-arg "BASE=${TEST_IMG_BASE}" --build-arg "VER_CUDA=${IMG_CUDA_VERSION}" --build-arg "VER_UBUNTU=${IMG_UBUNTU}" docker

    docker build -t ${DOCS_IMAGE} -f docker/Dockerfile.docs.deps --build-arg "BASE=${TEST_IMAGE}" --build-arg "VER_CUDA=${IMG_CUDA_VERSION}" --build-arg "VER_UBUNTU=${IMG_UBUNTU}" docker
done
