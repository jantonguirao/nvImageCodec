#!/bin/bash -ex

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
export VERSION=${VERSION:-6}
export REGISTRY_PREFIX=${REGISTRY_PREFIX:-"gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/"}
export PLATFORM=${PLATFORM:-"linux/amd64"}  # or "linux/arm64"
export ARCH=${ARCH:-"x86_64"}  # or "aarch64"

docker buildx create --name nvimagecodec_builder || echo "nvimagecodec_build already created"
docker buildx use nvimagecodec_builder
docker buildx inspect --bootstrap

####### BASE IMAGES #######

# Manylinux2014 with GCC 9
export MANYLINUX_GCC9="${REGISTRY_PREFIX}manylinux2014_${ARCH}.gcc9:v${VERSION}"
docker buildx build \
    --cache-from ${MANYLINUX_GCC9} --pull -t ${MANYLINUX_GCC9} \
    -f docker/Dockerfile.gcc9 \
    --build-arg "FROM_IMAGE_NAME=quay.io/pypa/manylinux2014_${ARCH}" \
    --platform ${PLATFORM} \
    --push \
    docker

# Manylinux2014 with GCC 10
export MANYLINUX_GCC10="${REGISTRY_PREFIX}manylinux2014_${ARCH}.gcc10:v${VERSION}"
docker buildx build \
    --cache-from ${MANYLINUX_GCC10} --pull -t ${MANYLINUX_GCC10} \
    -f docker/Dockerfile.gcc10 \
    --build-arg "FROM_IMAGE_NAME=quay.io/pypa/manylinux2014_${ARCH}" \
    --platform ${PLATFORM} \
    --push \
    docker

# CUDA 11.8.0
export CUDA_118="${REGISTRY_PREFIX}cuda11.8-${ARCH}:v${VERSION}"
docker buildx build \
    --cache-from ${CUDA_118} --pull -t ${CUDA_118} \
    -f docker/Dockerfile.cuda118.${ARCH}.deps \
    --platform ${PLATFORM} \
    --push \
    docker

# CUDA 12.3.0
export CUDA_123="${REGISTRY_PREFIX}cuda12.3-${ARCH}:v${VERSION}"
docker buildx build \
    --cache-from ${CUDA_123} --pull -t ${CUDA_123} \
    -f docker/Dockerfile.cuda123.${ARCH}.deps \
    --platform ${PLATFORM} \
    --push \
    docker

# GCC 9 (minimum supported)
export DEPS_GCC9="${REGISTRY_PREFIX}nvimgcodec_deps-${ARCH}-gcc9:v${VERSION}"
docker buildx build \
    --cache-from ${DEPS_GCC9} --pull -t ${DEPS_GCC9} \
    -f docker/Dockerfile.deps \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC9}" \
    --build-arg "ARCH=${ARCH}" \
    --platform ${PLATFORM} \
    --push \
    docker

# GCC 10
export DEPS_GCC10="${REGISTRY_PREFIX}nvimgcodec_deps-${ARCH}:v${VERSION}"
docker buildx build \
    --cache-from ${DEPS_GCC10} --pull -t ${DEPS_GCC10} \
    -f docker/Dockerfile.deps \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC10}" \
    --build-arg "ARCH=${ARCH}" \
    --platform ${PLATFORM} \
    --push \
    docker

####### BUILDER IMAGES #######

# GCC 9 (minimum supported), CUDA 11.8
export BUILDER_GCC9_CUDA_118="${REGISTRY_PREFIX}builder-cuda-11.8-gcc9-${ARCH}:v${VERSION}"
docker buildx build \
    --cache-from ${BUILDER_GCC9_CUDA_118} --pull -t ${BUILDER_GCC9_CUDA_118} \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=${DEPS_GCC9}" \
    --build-arg "CUDA_IMAGE=${CUDA_118}" \
    --platform ${PLATFORM} \
    --push \
    docker

# GCC 10, CUDA 11.8
export BUILDER_CUDA_118="${REGISTRY_PREFIX}builder-cuda-11.8-${ARCH}:v${VERSION}"
docker buildx build \
    --cache-from ${BUILDER_CUDA_118} --pull -t ${BUILDER_CUDA_118} \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=${DEPS_GCC10}" \
    --build-arg "CUDA_IMAGE=${CUDA_118}" \
    --platform ${PLATFORM} \
    --push \
    docker

# GCC 10, CUDA 12.3
export BUILDER_CUDA_123="${REGISTRY_PREFIX}builder-cuda-12.3-${ARCH}:v${VERSION}"
docker buildx build \
    --cache-from ${BUILDER_CUDA_123} --pull -t ${BUILDER_CUDA_123} \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=${DEPS_GCC10}" \
    --build-arg "CUDA_IMAGE=${CUDA_123}" \
    --platform ${PLATFORM} \
    --push \
    docker

####### TEST IMAGES #######

# CUDA 11.3 (minimum supported)
export RUNNER_CUDA_113="${REGISTRY_PREFIX}runner-linux-${ARCH}-cuda-11.3:v${VERSION}"
docker buildx build \
    --cache-from ${RUNNER_CUDA_113} --pull -t ${RUNNER_CUDA_113} \
    -f docker/Dockerfile.${ARCH} \
    --build-arg "BASE=nvidia/cuda:11.3.1-runtime-ubuntu20.04" \
    --build-arg "VER_CUDA=11.3.1" \
    --build-arg "VER_UBUNTU=20.04" \
    --platform ${PLATFORM} \
    --push \
    docker

# CUDA 11.8
export RUNNER_CUDA_118="${REGISTRY_PREFIX}runner-linux-${ARCH}-cuda-11.8:v${VERSION}"
docker buildx build \
    --cache-from ${RUNNER_CUDA_118} --pull -t ${RUNNER_CUDA_118} \
    -f docker/Dockerfile.${ARCH} \
    --build-arg "BASE=nvidia/cuda:11.8.0-runtime-ubuntu20.04" \
    --build-arg "VER_CUDA=11.8.0" \
    --build-arg "VER_UBUNTU=20.04" \
    --platform ${PLATFORM} \
    --push \
    docker

# CUDA 12.1
export RUNNER_CUDA_121="${REGISTRY_PREFIX}runner-linux-${ARCH}-cuda-12.1:v${VERSION}"
docker buildx build \
    --cache-from ${RUNNER_CUDA_121} --pull -t ${RUNNER_CUDA_121} \
    -f docker/Dockerfile.${ARCH} \
    --build-arg "BASE=nvidia/cuda:12.1.1-runtime-ubuntu20.04" \
    --build-arg "VER_CUDA=12.1.1" \
    --build-arg "VER_UBUNTU=20.04" \
    --platform ${PLATFORM} \
    --push \
    docker
