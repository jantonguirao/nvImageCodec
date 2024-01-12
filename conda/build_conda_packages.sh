#!/bin/bash

set -o xtrace
set -e

if [ -z $PYVER ]; then
    echo "PYVER is not set"
    exit 1
fi

export CONDA_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export ROOT_DIR="${CONDA_DIR}/.."
export VERSION=$(grep -Po 'set\(NVIMGCODEC_VERSION "\K[^"]*' ${ROOT_DIR}/CMakeLists.txt)
export BUILD_FLAVOR=${BUILD_FLAVOR:-""}  # nightly, weekly
export GIT_SHA=$(git rev-parse HEAD)
export TIMESTAMP=$(date +%Y%m%d)
export VERSION_SUFFIX=$(if [ "${BUILD_FLAVOR}" != "" ]; then \
                          echo .${BUILD_FLAVOR}.${TIMESTAMP}; \
                        fi)
export BUILD_VERSION=${VERSION}${VERSION_SUFFIX}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
export GIT_LFS_SKIP_SMUDGE=1
export CUDA_VERSION=$(echo $(ls /usr/local/cuda/lib64/libcudart.so*)  | sed 's/.*\.\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\)/\1.\2/')
export CONDA_OVERRIDE_CUDA=${CUDA_VERSION}

CONDA_BUILD_OPTIONS="--python=${PYVER} --exclusive-config-file config/conda_build_config.yaml"
CONDA_PREFIX=${CONDA_PREFIX:-$(dirname $CONDA_EXE)/..}

# Adding conda-forge channel for dependencies
conda config --add channels conda-forge
conda config --add channels nvidia
conda config --add channels local

conda build ${CONDA_BUILD_OPTIONS} recipe

# Copying the artifacts from conda prefix
mkdir -p ${ROOT_DIR}/artifacts
cp ${CONDA_PREFIX}/conda-bld/*/nvidia-nvimagecodec*.tar.bz2 ${ROOT_DIR}/artifacts
